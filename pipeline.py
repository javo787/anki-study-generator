"""
pipeline.py — Main orchestration pipeline
Connects: chunker → generator → filter → quality → output
"""

import csv
import os
import re
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field

# ── IMPORTS ────────────────────────────────────────────────────────────────────

try:
    from google import genai as google_genai
except ImportError:
    sys.exit("❌  Run: pip install google-genai")

try:
    from groq import Groq
except ImportError:
    sys.exit("❌  Run: pip install groq")

from chunker   import chunk_youtube, chunk_pdf, Chunk
from anki_gen  import extract_video_id, get_transcript_entries, build_proxy_config
from generator import (
    AIClient,
    GenerationResult,
    build_gemini_client,
    build_groq_client,
    build_openrouter_client,
    build_nvidia_client,
    generate_from_youtube,
    generate_from_pdf,
)
from filter    import filter_cards, get_kept_cards
from quality   import improve_cards, score_cards
from format_pass import format_cards
from usage     import record, is_available, print_usage, get_all_usage
from sources   import is_valid_time, restore_times, source_html
from profiles  import normalize_profile
from config    import load_config

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "output"

# ── DATA CLASSES ───────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All settings for one pipeline run."""
    source:       str          # "youtube" | "pdf"
    input:        str          # URL or file path
    deck:         str
    lang:         str  = "en"
    page_from:    int  = 1
    page_to:      int  = 99999
    run_filter:   bool = True   # Step 5: Groq filtering
    run_quality:  bool = True   # Step 6: Gemini Pro quality
    detect_topics: bool = True  # Step 3: topic detection
    on_progress:  object = None # callback for web UI
    source_links: bool = True   # YouTube: link every card back to its moment in the video
    profile:      str  = "medical"  # "medical" | "german" — what kind of material this is (see profiles.py)


@dataclass
class PipelineResult:
    cards:          list[dict]
    filename:       str
    total_raw:      int   = 0
    total_filtered: int   = 0
    total_final:    int   = 0
    duration_sec:   float = 0.0
    stages:         dict  = field(default_factory=dict)
    errors:         list  = field(default_factory=list)

    def summary(self) -> str:
        mins = self.duration_sec / 60
        return (
            f"✅  Pipeline complete in {mins:.1f} min\n"
            f"   Raw cards:      {self.total_raw}\n"
            f"   After filter:   {self.total_filtered}\n"
            f"   Final cards:    {self.total_final}\n"
            f"   Saved to:       {self.filename}"
        )


# ── CLIENT BUILDER ─────────────────────────────────────────────────────────────

def build_clients(cfg: dict) -> list[AIClient]:
    """
    Build all available AI clients from config.
    Returns sorted list by priority — best first.
    """
    clients = []

    # Gemini Flash — primary generator (priority 1)
    if cfg.get("gemini_api_key"):
        try:
            clients.append(build_gemini_client(
                api_key  = cfg["gemini_api_key"],
                model    = "gemini-2.5-flash",
                priority = 1,
            ))
        except Exception as e:
            print(f"  ⚠️  Gemini Flash setup failed: {e}")

    # OpenRouter — first fallback (priority 2)
    if cfg.get("openrouter_api_key"):
        try:
            clients.append(build_openrouter_client(cfg["openrouter_api_key"]))
        except Exception as e:
            print(f"  ⚠️  OpenRouter setup failed: {e}")

    # NVIDIA — fallback when Gemini 503 (priority 2)
    if cfg.get("nvidia_api_key"):
        try:
            clients.append(build_nvidia_client(cfg["nvidia_api_key"]))
            clients[-1].priority = 2
        except Exception as e:
            print(f"  ⚠️  NVIDIA setup failed: {e}")

    if cfg.get("groq_api_key"):
        try:
            clients.append(build_groq_client(cfg["groq_api_key"]))
            clients[-1].priority = 3
        except Exception as e:
            print(f"  ⚠️  Groq setup failed: {e}")

    if not clients:
        raise RuntimeError("No AI clients available. Check API keys in config.json.")

    print(f"  🤖  Available AI: {[c.name for c in clients]}")
    return clients


# ── OUTPUT ─────────────────────────────────────────────────────────────────────

def _safe_name(text: str) -> str:
    return "".join(
        c if c.isalnum() or c in "-_ " else ""
        for c in text
    ).strip().replace(" ", "_")


def _as_text(value) -> str:
    """Raw LLM output can give a field as a list of lines — flatten it."""
    if isinstance(value, list):
        value = "\n".join(str(x) for x in value)
    return str(value or "")


def _clean_field(text) -> str:
    """One card field -> single-line text for the import file (newline -> <br>)."""
    text = _as_text(text).replace("\t", " ").replace("\r\n", "\n")
    return text.replace("\n", "<br>")


def _clean_tags(tags) -> str:
    """Anki tags are space-separated, so a tag containing a space would silently
    become two tags. Join words with '_' and drop empties/duplicates."""
    if isinstance(tags, str):
        tags = re.split(r"[,;]+", tags)
    seen = []
    for t in tags or []:
        t = re.sub(r"\s+", "_", str(t).strip())
        if t and t not in seen:
            seen.append(t)
    return " ".join(seen)


def save_cards(cards: list[dict], deck: str, filename_suffix: str = "", video_id: str | None = None) -> str:
    """
    Save cards to output folder in Anki import format.

    One file holds both note types. The note type is read from the FIRST COLUMN
    (`#notetype column:1`) instead of from header lines: Anki only honours '#'
    header lines at the very top of the file, so a second `#notetype:Cloze`
    block in the middle was ignored and every cloze card was imported as a
    Basic note showing raw {{c1::...}} text.

    Every field is CSV-quoted. With `#separator:tab` Anki treats a field that
    starts with `"` as quoted, which used to strip the quotes from the text.

    filename_suffix: optional tag appended to the filename only (e.g. "RAW")
    — the #deck: header always uses the clean `deck` name so the file still
    imports into the right Anki deck if opened directly.

    video_id: when given, a card that has a valid time `t` (seconds, see sources.py)
    gets a "▶ mm:ss" link to that moment of the video appended to its back.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    suffix   = f"_{_safe_name(filename_suffix)}" if filename_suffix else ""
    filename = os.path.join(OUTPUT_DIR, f"anki_{_safe_name(deck)}{suffix}_{date_str}.txt")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write("#separator:tab\n")
        f.write("#html:true\n")
        f.write("#notetype column:1\n")
        f.write(f"#deck:{' '.join(str(deck).split())}\n")
        f.write("#tags column:4\n")

        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_ALL, lineterminator="\n")
        for card in cards:
            is_cloze = str(card.get("type", "basic")).strip().lower() == "cloze"
            front    = _as_text(card.get("front", ""))
            if is_cloze:
                # Only legacy "___ (answer)" cards get changed; see the guard
                # inside _convert_to_anki_cloze.
                front = _convert_to_anki_cloze(front)
            back = _clean_field(card.get("back", ""))
            if video_id and is_valid_time(card.get("t")):
                link = source_html(video_id, card["t"])
                back = f"{back}<br><br>{link}" if back else link
            writer.writerow([
                "Cloze" if is_cloze else "Basic",
                _clean_field(front),
                back,
                _clean_tags(card.get("tags", [])),
            ])

    return filename


_REAL_CLOZE = re.compile(r"\{\{c\d+::")


def _convert_to_anki_cloze(text: str) -> str:
    """
    Convert our legacy cloze format to Anki cloze format.

    Input:  "The ___ drains CSF into venous sinuses. (Arachnoid villi)"
    Output: "The {{c1::Arachnoid villi}} drains CSF into venous sinuses."

    Text that already contains real {{cN::...}} markup is returned unchanged.
    Without this guard a valid card such as
    "{{c1::Choroid plexus}} (in the ventricles)" was corrupted into two blanks.
    """
    if _REAL_CLOZE.search(text):
        return text

    # Extract answer from parentheses at end
    match = re.search(r"\(([^)]+)\)\s*$", text)
    if not match:
        return text  # already in another format or no answer found

    answer   = match.group(1).strip()
    question = text[:match.start()].strip()

    # Replace ___ with {{c1::answer}}
    if "___" in question:
        return question.replace("___", f"{{{{c1::{answer}}}}}")

    # No blank found — wrap answer at end
    return f"{question} {{{{c1::{answer}}}}}"


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def _usage_key(provider_name: str):
    """Map an AIClient name such as 'Gemini (gemini-2.5-flash)' to the key the
    quota monitor (usage.LIMITS) displays. Names were previously lower-cased into
    keys like 'gemini_gemini-2.5-flash' that the monitor never showed."""
    n = (provider_name or "").lower()
    if n in ("", "none"):
        return None
    if "gemini" in n:
        return "gemini_pro" if "pro" in n else "gemini_flash"
    for known in ("groq", "openrouter", "cerebras"):
        if known in n:
            return known
    return re.sub(r"[^a-z0-9_-]+", "_", n).strip("_")


def run(config: PipelineConfig, on_progress: callable = None) -> PipelineResult:
    """
    Run the full pipeline end to end.

    Steps:
      1. Get source (transcript or PDF)
      2. Smart chunking
      3. Topic detection per chunk
      4. Card generation with AI fallback  → saves a RAW pre-filter .txt
      5. Groq filtering (optional)
      6. Gemini Pro quality control (optional)
      6.5. Format pass — real Anki cloze syntax, markdown → HTML (format_pass.py)
      7. Score and sort
      8. Save to file
    """
    profile   = normalize_profile(config.profile)      # fail early on a typo, not after the API calls
    cfg       = load_config()
    started   = time.time()
    result    = PipelineResult(cards=[], filename="")
    result.stages["profile"] = profile
    errors    = []

    print(f"\n{'─'*50}")
    print(f"  🧠  Pipeline starting")
    print(f"  Source : {config.source.upper()} — {config.input[:60]}")
    print(f"  Deck   : {config.deck}")
    print(f"  Lang   : {config.lang}")
    print(f"{'─'*50}\n")

    # ── Validate config ────────────────────────────────────────────────────────
    if not cfg.get("gemini_api_key") and not cfg.get("openrouter_api_key"):
        raise RuntimeError("No generation API key found. Add gemini_api_key or openrouter_api_key to config.json.")

    # ── Build clients ──────────────────────────────────────────────────────────
    clients = build_clients(cfg)

    # ── Step 1-4: Generate raw cards ───────────────────────────────────────────
    print("  📥  STEP 1-4: Fetching + Chunking + Topic detection + Generation\n")

    def _progress(chunk_idx, chunk_total, topic):
        if on_progress:
            on_progress({
                "stage":       "generating",
                "chunk":       chunk_idx,
                "total":       chunk_total,
                "topic":       topic,
                "pct":         round(chunk_idx / chunk_total * 40),
            })

    try:
        if config.source == "youtube":
            gen_result = generate_from_youtube(
                url           = config.input,
                deck          = config.deck,
                clients       = clients,
                detect_topics = config.detect_topics,
                on_progress   = _progress,
                lang          = config.lang,
                proxy_config  = build_proxy_config(cfg),
                profile       = profile,
            )
        elif config.source == "pdf":
            gen_result = generate_from_pdf(
                path          = config.input,
                deck          = config.deck,
                clients       = clients,
                page_from     = config.page_from,
                page_to       = config.page_to,
                detect_topics = config.detect_topics,
                on_progress   = _progress,
                lang          = config.lang,
                profile       = profile,
            )
        elif config.source == "video":
            from video_gen import transcribe_local_video
            import re

            def _video_progress(step, total, msg):
                if on_progress:
                    on_progress({
                        "stage": "generating",
                        "topic": msg,
                        "pct":   round(step / total * 20),
                    })

            # Step 1: transcribe
            transcript = transcribe_local_video(
                video_path   = config.input,
                groq_api_key = cfg.get("groq_api_key", ""),
                language     = config.lang,
                on_progress  = _video_progress,
            )

            # Save the raw transcript to disk — useful for research/debugging,
            # and lets you re-run card generation later without re-transcribing.
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            transcript_path = os.path.join(
                OUTPUT_DIR,
                f"transcript_{_safe_name(config.deck)}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt",
            )
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"  💾  Transcript saved: {transcript_path}")

            # Step 2: split transcript into chunks by character count
            CHUNK_SIZE = 5000  # ~600 words per chunk → ~4 min of speech
            
            sentences  = re.split(r'(?<=[.!?])\s+', transcript)
            raw_chunks = []
            current    = []
            current_len = 0

            for sentence in sentences:
                if current_len + len(sentence) > CHUNK_SIZE and current:
                    raw_chunks.append(" ".join(current))
                    current, current_len = [], 0
                current.append(sentence)
                current_len += len(sentence)
            if current:
                raw_chunks.append(" ".join(current))

            # Step 3: wrap as Chunk objects
            from chunker import Chunk
            total_chunks = len(raw_chunks)
            chunks = [
                Chunk(
                    text       = text,
                    index      = i + 1,
                    total      = total_chunks,
                    source     = "youtube",  # reuse youtube pipeline
                    start_time = i * 240.0,
                    end_time   = (i + 1) * 240.0,
                )
                for i, text in enumerate(raw_chunks)
            ]

            print(f"  📦  {len(chunks)} chunks from video transcript")

            from generator import _run_pipeline
            gen_result = _run_pipeline(chunks, config.deck, clients, config.detect_topics, _progress, config.lang, profile)

    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")

    raw_cards           = gen_result.cards
    result.total_raw    = len(raw_cards)

    chunks_failed = gen_result.chunks_total - gen_result.chunks_done
    coverage_pct  = (round(gen_result.chunks_done / gen_result.chunks_total * 100)
                     if gen_result.chunks_total else 0)
    result.stages["generation"] = {
        "cards":         result.total_raw,
        "chunks":        gen_result.chunks_done,
        "chunks_total":  gen_result.chunks_total,
        "chunks_failed": chunks_failed,
        "coverage_pct":  coverage_pct,
        "ai_used":       gen_result.ai_used,
    }

    # A source that is only partly covered must never look like a clean success:
    # the cards from the failed parts are simply absent from the deck.
    if chunks_failed:
        errors.append(
            f"Only {gen_result.chunks_done} of {gen_result.chunks_total} parts of the source "
            f"produced cards ({coverage_pct}% coverage) — the rest is MISSING from the deck."
        )
        errors.extend(gen_result.errors)

    video_id = None
    if config.source == "youtube" and config.source_links:
        try:
            video_id = extract_video_id(config.input)
        except ValueError:
            video_id = None

    # Track usage (under the keys the quota monitor actually displays)
    for provider, count in gen_result.ai_used.items():
        provider_key = _usage_key(provider)
        if provider_key:
            for _ in range(count):
                record(provider_key, tokens=2000)

    if on_progress:
        on_progress({"stage": "generated", "cards": result.total_raw, "pct": 40})

    print(f"\n  ✅  Step 1-4 complete: {result.total_raw} raw cards\n")

    # Save the raw, pre-filter cards to their own file — useful for research:
    # lets you see exactly what the filter step is removing (and why, if you
    # compare against the final file's card count).
    if raw_cards:
        raw_path = save_cards(raw_cards, config.deck, filename_suffix="RAW_prefilter", video_id=video_id)
        print(f"  💾  Raw pre-filter cards saved: {raw_path}")

    # Let the per-minute Gemini limit recover before the next step — but only when
    # that step would hit the SAME limit. The filter runs on Groq, and quality runs on
    # gemini-2.5-pro (its own per-model quota) when a Pro key is set. Without a Pro key
    # quality falls back to Flash on the generation key (see Step 6), so wait then.
    # (This used to be an unconditional 60 s pause on every run.)
    quality_reuses_flash = (
        config.run_quality
        and not cfg.get("gemini_pro_key")
        and bool(cfg.get("gemini_api_key"))
        and result.total_raw > 0
        and any("gemini" in provider.lower() for provider in gen_result.ai_used)
    )
    if quality_reuses_flash:
        print("  ⏸️   Waiting 60s for Gemini RPM limit to recover (quality reuses Flash)...")
        time.sleep(60)

    # ── Step 5: Filter ─────────────────────────────────────────────────────────
    filtered_cards = raw_cards

    if config.run_filter and cfg.get("groq_api_key") and raw_cards:
        print("  🔎  STEP 5: Groq filtering\n")

        if on_progress:
            on_progress({"stage": "filtering", "pct": 50})

        try:
            filter_result  = filter_cards(
                cards        = raw_cards,
                deck         = config.deck,
                groq_api_key = cfg["groq_api_key"],
                profile      = profile,
            )
            filtered_cards = get_kept_cards(filter_result)
            record("groq", tokens=len(raw_cards) * 150)

            result.stages["filter"] = {
                "cards_in":  filter_result.cards_in,
                "cards_out": filter_result.cards_out,
                "reasons":   filter_result.reasons,
            }
            print(f"\n  ✅  Step 5 complete: {len(filtered_cards)} cards kept\n")

        except Exception as e:
            print(f"  ⚠️  Filter step failed: {e} — skipping")
            errors.append(f"Filter: {e}")
            filtered_cards = raw_cards

    result.total_filtered = len(filtered_cards)

    if on_progress:
        on_progress({"stage": "filtered", "cards": result.total_filtered, "pct": 70})

    # ── Step 6: Quality control ────────────────────────────────────────────────
    final_cards = filtered_cards

    if config.run_quality and cfg.get("gemini_pro_key") and filtered_cards:
        print("  ✨  STEP 6: Gemini Pro quality control\n")

        if on_progress:
            on_progress({"stage": "quality", "pct": 80})

        try:
            final_cards, quality_result = improve_cards(
                cards   = filtered_cards,
                deck    = config.deck,
                api_key = cfg["gemini_pro_key"],
                use_pro = True,
                lang    = config.lang,
                profile = profile,
            )
            record("gemini_pro", tokens=len(filtered_cards) * 200)

            result.stages["quality"] = {
                "mnemonics_added":   quality_result.mnemonics_added,
                "phrasing_improved": quality_result.phrasing_improved,
                "model_used":        quality_result.model_used,
            }
            print(f"\n  ✅  Step 6 complete\n")

        except Exception as e:
            print(f"  ⚠️  Quality step failed: {e} — skipping")
            errors.append(f"Quality: {e}")
            final_cards = filtered_cards

    elif config.run_quality and cfg.get("gemini_api_key") and filtered_cards:
        # Fallback to Flash if no Pro key
        print("  ✨  STEP 6: Gemini Flash quality control (no Pro key)\n")
        try:
            final_cards, quality_result = improve_cards(
                cards   = filtered_cards,
                deck    = config.deck,
                api_key = cfg["gemini_api_key"],
                use_pro = False,
                lang    = config.lang,
                profile = profile,
            )
        except Exception as e:
            print(f"  ⚠️  Quality step failed: {e} — skipping")
            final_cards = filtered_cards

    # ── Step 6.5: Format pass ──────────────────────────────────────────────────
    # Fixes real Anki cloze syntax ({{c1::...}}), converts markdown to HTML,
    # and locally validates/repairs every cloze card so a broken one can never
    # reach the saved file — see format_pass.py for why this is a separate step.
    if final_cards:
        print("  🎨  STEP 6.5: Format pass\n")

        if on_progress:
            on_progress({"stage": "formatting", "pct": 90})

        try:
            final_cards, format_result = format_cards(
                cards        = final_cards,
                deck         = config.deck,
                groq_api_key = cfg.get("groq_api_key", ""),
                lang         = config.lang,
                profile      = profile,
            )
            result.stages["format"] = {
                "cloze_fixed":       format_result.cloze_fixed,
                "cloze_demoted":     format_result.cloze_demoted,
                "markdown_stripped": format_result.markdown_stripped,
                "uz_transliterated": format_result.uz_transliterated,
            }
            print(f"\n  ✅  Step 6.5 complete\n")
        except Exception as e:
            print(f"  ⚠️  Format step failed: {e} — skipping")
            errors.append(f"Format: {e}")

    # ── Step 7: Score and sort ─────────────────────────────────────────────────
    # The score is a medical heuristic (rewards bullets, 💡, digits, "which/where" questions)
    # and re-orders the deck by it. A textbook deck keeps the book's own order: that is
    # the order the learner meets the words in, and new cards are shown in import order.
    if profile != "german":
        final_cards = score_cards(final_cards, lang=config.lang)

    # ── Step 8: Save ───────────────────────────────────────────────────────────
    # The filter / quality / format passes let a model rewrite the cards and it may
    # drop fields it does not know, so the video time is matched back to the original
    # cards (sources.restore_times) before the links are written.
    if video_id:
        restored = restore_times(final_cards, raw_cards)
        if restored:
            print(f"  🔗  Source time restored on {restored} rewritten card(s)")
    filename = save_cards(final_cards, config.deck, video_id=video_id)

    result.cards       = final_cards
    result.filename    = filename
    result.total_final = len(final_cards)
    result.duration_sec = time.time() - started
    result.errors      = errors

    if on_progress:
        on_progress({
            "stage":    "done",
            "cards":    result.total_final,
            "filename": os.path.basename(filename),
            "pct":      100,
        })

    print(f"\n{'─'*50}")
    print(f"  {result.summary()}")
    print(f"{'─'*50}\n")

    print_usage()
    return result


# ── CONVENIENCE WRAPPERS ───────────────────────────────────────────────────────

def run_youtube(
    url:          str,
    deck:         str,
    lang:         str  = "en",
    run_filter:   bool = True,
    run_quality:  bool = True,
    on_progress:  callable = None,
    source_links: bool = True,
    profile:      str  = "medical",
) -> PipelineResult:
    return run(PipelineConfig(
        source       = "youtube",
        input        = url,
        deck         = deck,
        lang         = lang,
        run_filter   = run_filter,
        run_quality  = run_quality,
        on_progress  = on_progress,
        source_links = source_links,
        profile      = profile,
    ), on_progress=on_progress)


def run_pdf(
    path:         str,
    deck:         str,
    lang:         str  = "en",
    page_from:    int  = 1,
    page_to:      int  = 99999,
    run_filter:   bool = True,
    run_quality:  bool = True,
    on_progress:  callable = None,
    profile:      str  = "medical",
) -> PipelineResult:
    return run(PipelineConfig(
        source      = "pdf",
        input       = path,
        deck        = deck,
        lang        = lang,
        page_from   = page_from,
        page_to     = page_to,
        run_filter  = run_filter,
        run_quality = run_quality,
        on_progress = on_progress,
        profile     = profile,
    ), on_progress=on_progress)


def run_video(
    path:         str,
    deck:         str,
    lang:         str  = "en",
    run_filter:   bool = True,
    run_quality:  bool = True,
    on_progress:  callable = None,
    profile:      str  = "medical",
) -> PipelineResult:
    return run(PipelineConfig(
        source      = "video",
        input       = path,
        deck        = deck,
        lang        = lang,
        run_filter  = run_filter,
        run_quality = run_quality,
        on_progress = on_progress,
        profile     = profile,
    ), on_progress=on_progress)


# ── DEBUG ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python pipeline.py youtube <url> <deck>")
        print("  python pipeline.py pdf <path> <deck> [from] [to]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "youtube":
        url  = sys.argv[2]
        deck = sys.argv[3] if len(sys.argv) > 3 else "Neuroanatomy"
        run_youtube(url, deck)

    elif mode == "pdf":
        path      = sys.argv[2]
        deck      = sys.argv[3] if len(sys.argv) > 3 else "My Deck"
        page_from = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        page_to   = int(sys.argv[5]) if len(sys.argv) > 5 else 99999
        run_pdf(path, deck, page_from=page_from, page_to=page_to)

    elif mode == "video":
        path      = sys.argv[2]
        deck      = sys.argv[3] if len(sys.argv) > 3 else "My Deck"
        run_video(path, deck)