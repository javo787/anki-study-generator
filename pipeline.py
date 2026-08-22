"""
pipeline.py — Main orchestration pipeline
Connects: chunker → generator → filter → quality → output
"""

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


def save_cards(cards: list[dict], deck: str, filename_suffix: str = "") -> str:
    """
    Save cards to output folder in Anki import format.
    Handles both Basic and Cloze note types in one file.

    filename_suffix: optional tag appended to the filename only (e.g. "RAW")
    — the #deck: header always uses the clean `deck` name so the file still
    imports into the right Anki deck if opened directly.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    suffix   = f"_{_safe_name(filename_suffix)}" if filename_suffix else ""
    filename = os.path.join(OUTPUT_DIR, f"anki_{_safe_name(deck)}{suffix}_{date_str}.txt")

    basic_cards = [c for c in cards if c.get("type", "basic") == "basic"]
    cloze_cards = [c for c in cards if c.get("type") == "cloze"]

    lines = []

    # ── Basic cards ────────────────────────────────────────────────────────────
    if basic_cards:
        lines += [
            "#separator:tab",
            "#html:true",
            f"#deck:{deck}",
            "#notetype:Basic",
            "#columns:Front\tBack\tTags",
        ]
        for card in basic_cards:
            front = card["front"].replace("\t", " ").replace("\n", "<br>")
            back  = card["back"].replace("\t", " ").replace("\n", "<br>")
            tags  = " ".join(card.get("tags", []))
            lines.append(f"{front}\t{back}\t{tags}")

    # ── Cloze cards ────────────────────────────────────────────────────────────
    if cloze_cards:
        lines += [
            "",
            "#notetype:Cloze",
            "#columns:Text\tBack Extra\tTags",
        ]
        for card in cloze_cards:
            # Convert "The ___ drains CSF. (Arachnoid villi)"
            # to Anki cloze: "The {{c1::Arachnoid villi}} drains CSF."
            front = _convert_to_anki_cloze(card["front"])
            back  = card["back"].replace("\t", " ").replace("\n", "<br>")
            tags  = " ".join(card.get("tags", []))
            lines.append(f"{front}\t{back}\t{tags}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filename


def _convert_to_anki_cloze(text: str) -> str:
    """
    Convert our cloze format to Anki cloze format.

    Input:  "The ___ drains CSF into venous sinuses. (Arachnoid villi)"
    Output: "The {{c1::Arachnoid villi}} drains CSF into venous sinuses."
    """
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
    cfg       = load_config()
    started   = time.time()
    result    = PipelineResult(cards=[], filename="")
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
            gen_result = _run_pipeline(chunks, config.deck, clients, config.detect_topics, _progress, config.lang)

    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}")

    raw_cards           = gen_result.cards
    result.total_raw    = len(raw_cards)
    result.stages["generation"] = {
        "cards":      result.total_raw,
        "chunks":     gen_result.chunks_done,
        "ai_used":    gen_result.ai_used,
    }

    # Track usage
    for provider, count in gen_result.ai_used.items():
        provider_key = provider.lower().replace(" ", "_").replace("(", "").replace(")", "")
        for _ in range(count):
            record(provider_key, tokens=2000)

    if on_progress:
        on_progress({"stage": "generated", "cards": result.total_raw, "pct": 40})

    print(f"\n  ✅  Step 1-4 complete: {result.total_raw} raw cards\n")

    # Save the raw, pre-filter cards to their own file — useful for research:
    # lets you see exactly what the filter step is removing (and why, if you
    # compare against the final file's card count).
    if raw_cards:
        raw_path = save_cards(raw_cards, config.deck, filename_suffix="RAW_prefilter")
        print(f"  💾  Raw pre-filter cards saved: {raw_path}")

    # Пауза чтобы Gemini RPM лимит восстановился перед quality шагом
    if result.total_raw > 0:
        print("  ⏸️   Waiting 60s for Gemini RPM limit to recover...")
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
    final_cards = score_cards(final_cards, lang=config.lang)

    # ── Step 8: Save ───────────────────────────────────────────────────────────
    filename = save_cards(final_cards, config.deck)

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
) -> PipelineResult:
    return run(PipelineConfig(
        source      = "youtube",
        input       = url,
        deck        = deck,
        lang        = lang,
        run_filter  = run_filter,
        run_quality = run_quality,
        on_progress = on_progress,
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
    ), on_progress=on_progress)


def run_video(
    path:         str,
    deck:         str,
    lang:         str  = "en",
    run_filter:   bool = True,
    run_quality:  bool = True,
    on_progress:  callable = None,
) -> PipelineResult:
    return run(PipelineConfig(
        source      = "video",
        input       = path,
        deck        = deck,
        lang        = lang,
        run_filter  = run_filter,
        run_quality = run_quality,
        on_progress = on_progress,
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