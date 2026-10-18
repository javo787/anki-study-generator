"""
generator.py — Card generation orchestrator
Combines smart chunking + multi-AI fallback + topic detection
Requires: pip install google-genai groq
"""

import re
import sys
import json
import time
from dataclasses import dataclass, field

from cards_json import parse_cards

# ── IMPORTS ────────────────────────────────────────────────────────────────────

try:
    from google import genai as google_genai
except ImportError:
    sys.exit("❌  Run: pip install google-genai")

try:
    from chunker import Chunk, chunk_youtube, chunk_pdf
except ImportError:
    sys.exit("❌  chunker.py not found in the same directory")

try:
    from anki_gen import (
        extract_video_id,
        get_transcript_entries,
        get_transcript,
        generate_cards as gemini_generate,
        deduplicate,
        build_client as build_gemini,
        build_language_instruction,
    )
except ImportError:
    sys.exit("❌  anki_gen.py not found in the same directory")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

GEMINI_FLASH_MODEL = "gemini-2.5-flash"
GEMINI_PRO_MODEL   = "gemini-2.5-pro"
GROQ_MODEL         = "llama-3.3-70b-versatile"
OPENROUTER_MODEL   = "meta-llama/llama-3.3-70b-instruct:free"

TOPIC_PROMPT = """\
In 4-6 words, name the main anatomical or medical topic of this text.
Examples: "CSF circulation and drainage", "Brachial plexus roots", "Heart valve anatomy"
Reply with the topic only — no punctuation, no explanation."""


# Rate limits (Gemini free tier is limited per MINUTE and per DAY).
MAX_RATE_LIMIT_WAIT  = 90   # seconds we are willing to wait when the API says "retry in N s"
SECOND_PASS_COOLDOWN = 45   # seconds before chunks that failed are given one more chance

_RETRY_DELAY = re.compile(r"retry(?:Delay)?['\"]?\s*(?:in|[:=])\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _is_daily_quota(err: str) -> bool:
    """A per-DAY quota will not recover by waiting a minute — go to a fallback."""
    return "PerDay" in err


def _retry_delay_seconds(err: str):
    """How long the API asked us to wait ('Please retry in 41.2s' / retryDelay '41s'), or None."""
    m = _RETRY_DELAY.search(err)
    return float(m.group(1)) if m else None


def _transcript_language_priority(lang: str) -> list[str]:
    """
    Which YouTube caption track(s) to look for, in priority order, for a
    given "Card language" choice. A native transcript in the target
    language gives the model real source material instead of making it
    translate mid-generation, so we ask for that first and fall back to
    English (get_transcript_entries falls back further still — see
    anki_gen.py — including translating whatever track IS available).
    """
    lang = (lang or "en").lower()
    if lang == "en":
        return ["en"]
    return [lang, "en"]

# ── DATA CLASSES ───────────────────────────────────────────────────────────────

@dataclass
class AIClient:
    name:     str
    provider: str   # "gemini" | "groq" | "openrouter" | "cerebras"
    model:    str
    client:   object
    priority: int   # lower = preferred


@dataclass
class GenerationResult:
    cards:        list[dict]
    chunks_total: int
    chunks_done:  int
    errors:       list[str] = field(default_factory=list)
    ai_used:      dict      = field(default_factory=dict)  # provider → count (successful chunks only)
    failed_chunks: list[int] = field(default_factory=list)  # chunk numbers that produced no cards

    @property
    def success_rate(self) -> float:
        if self.chunks_total == 0:
            return 0.0
        return self.chunks_done / self.chunks_total * 100


# ── AI CLIENT BUILDERS ─────────────────────────────────────────────────────────

def build_gemini_client(api_key: str, model: str, priority: int) -> AIClient:
    client = google_genai.Client(api_key=api_key)
    return AIClient(
        name=f"Gemini ({model})",
        provider="gemini",
        model=model,
        client=client,
        priority=priority,
    )


def build_groq_client(api_key: str) -> AIClient:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")
    return AIClient(
        name="Groq Llama",
        provider="groq",
        model=GROQ_MODEL,
        client=Groq(api_key=api_key),
        priority=3,
    )


def build_openrouter_client(api_key: str) -> AIClient:
    """OpenRouter uses OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return AIClient(
        name="OpenRouter Llama",
        provider="openrouter",
        model=OPENROUTER_MODEL,
        client=client,
        priority=4,
    )

def build_nvidia_client(api_key: str) -> AIClient:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )
    return AIClient(
        name     = "NVIDIA Llama",
        provider = "openrouter",
        model    = "meta/llama-3.3-70b-instruct",
        client   = client,
        priority = 2,
    )


# ── TOPIC DETECTION ────────────────────────────────────────────────────────────

def detect_topic(chunk: Chunk, gemini_client: AIClient) -> str:
    """
    Ask Gemini Flash to name the topic of a chunk in 4-6 words.
    Falls back to time label or page range if AI fails.
    """
    preview = chunk.text[:1500]
    try:
        response = gemini_client.client.models.generate_content(
            model=gemini_client.model,
            contents=f"{TOPIC_PROMPT}\n\nText:\n{preview}",
            config={"temperature": 0.1},
        )
        topic = response.text.strip().strip("\"'.,")
        return topic if topic else _fallback_topic(chunk)
    except Exception:
        return _fallback_topic(chunk)


def _fallback_topic(chunk: Chunk) -> str:
    if chunk.source == "youtube":
        return f"segment {chunk.time_label}"
    return f"pages {chunk.page_from}–{chunk.page_to}"


# ── CARD GENERATION PER CHUNK ──────────────────────────────────────────────────

GROQ_CARD_PROMPT = """\
You are an expert medical Anki card creator.

RULES:
1. One card = one testable fact
2. Front: specific question — What / Where / Which / How / Name / List
3. Back: clean bullet list (max 5 bullets) + 💡 mnemonic where useful
4. Generate both basic Q&A and cloze deletion cards
5. Cloze cards ("type": "cloze"): "front" MUST use real Anki cloze syntax, for example
   "The {{{{c1::dura mater}}}} is the outermost meningeal layer." — never underscore (___) placeholders
6. Number of cards follows the content: one card per distinct testable fact, up to 35.
   NEVER pad to reach a number — if the text has few testable facts, return only those few cards
7. Topic: {topic} | Chunk {chunk_index} of {chunk_total}
8. LANGUAGE: {language_instruction}
9. FORMATTING: HTML only, never markdown (no **bold** — use <b>bold</b>, and <br> for line breaks)
10. The text is an automatic speech-recognition transcript: it may lack punctuation and medical terms
    are often misspelled (e.g. "coroid plexus" = "choroid plexus"). Write every term with its correct
    standard spelling. Correct spelling only — never add or change facts the lecturer did not state.

OUTPUT: JSON only — no markdown, no explanation.
{{"cards": [{{"front": "...", "back": "...", "type": "basic", "tags": ["..."]}}]}}"""


def _parse_response(raw: str) -> list[dict]:
    """Parse the model reply into cards (tolerant: see cards_json.py)."""
    return parse_cards(raw)


def _generate_with_groq(chunk: Chunk, deck: str, ai: AIClient, lang: str = "en") -> list[dict]:
    """Generate cards using Groq API."""
    system = GROQ_CARD_PROMPT.format(
        topic=chunk.topic_hint or "general",
        chunk_index=chunk.index,
        chunk_total=chunk.total,
        language_instruction=build_language_instruction(lang),
    )
    prompt = (
        f"Deck: {deck}\n\n"
        f"Text:\n{chunk.text}\n\n"
        f"Generate Anki cards. JSON only."
    )
    try:
        response = ai.client.chat.completions.create(
            model=ai.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        # Use centralized robust parser
        return _parse_response(raw)
    except Exception as e:
        raise RuntimeError(f"Groq error: {e}")


def _generate_with_openrouter(chunk: Chunk, deck: str, ai: AIClient, lang: str = "en") -> list[dict]:
    """Generate cards using OpenRouter API (OpenAI-compatible)."""
    system = GROQ_CARD_PROMPT.format(
        topic=chunk.topic_hint or "general",
        chunk_index=chunk.index,
        chunk_total=chunk.total,
        language_instruction=build_language_instruction(lang),
    )
    prompt = (
        f"Deck: {deck}\n\n"
        f"Text:\n{chunk.text}\n\n"
        f"Generate Anki cards. JSON only."
    )
    try:
        response = ai.client.chat.completions.create(
            model=ai.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        # Use centralized robust parser
        return _parse_response(raw)
    except Exception as e:
        raise RuntimeError(f"OpenRouter error: {e}")


def generate_chunk(
    chunk:   Chunk,
    deck:    str,
    clients: list[AIClient],
    lang:    str = "en",
) -> tuple[list[dict], str]:

    sorted_clients   = sorted(clients, key=lambda c: c.priority)
    gemini_clients   = [c for c in sorted_clients if c.provider == "gemini"]
    fallback_clients = [c for c in sorted_clients if c.provider != "gemini"]

    # ── Step 1: Try Gemini up to 3 times ──────────────────────────────────────
    gemini_exhausted = False

    for attempt in range(1, 4):
        for ai in gemini_clients:
            try:
                cards = gemini_generate(
                    chunk       = chunk.text,
                    deck        = deck,
                    client      = ai.client,
                    chunk_index = chunk.index,
                    chunk_total = chunk.total,
                    topic       = chunk.topic_hint or "general",
                    lang        = lang,
                )
                if cards:
                    return cards, ai.name

            except Exception as e:
                err = str(e)

                if "503" in err or "overload" in err.lower():
                    if attempt < 3:
                        wait = attempt * 15
                        print(f"  ⏳  Gemini 503, attempt {attempt}/3, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  ⚠️  Gemini 503 x3 — switching to fallback...")
                        gemini_exhausted = True
                        break

                elif "429" in err or "RESOURCE_EXHAUSTED" in err:
                    delay = None if _is_daily_quota(err) else _retry_delay_seconds(err)
                    if delay is not None and delay <= MAX_RATE_LIMIT_WAIT and attempt < 3:
                        wait = delay + 1
                        print(f"  ⏳  Gemini 429 — API asked to wait {delay:.0f}s, "
                              f"attempt {attempt}/3...")
                        time.sleep(wait)
                    else:
                        why = "daily quota" if _is_daily_quota(err) else "rate limit"
                        print(f"  ⏳  Gemini 429 ({why}) — switching to fallback...")
                        gemini_exhausted = True
                        break

                else:
                    print(f"  ⚠️  Gemini error: {err[:60]} — switching to fallback...")
                    gemini_exhausted = True
                    break

        if gemini_exhausted:
            break

    # ── Step 2: Try fallback clients (NVIDIA → Groq → OpenRouter) ─────────────
    for ai in fallback_clients:
        try:
            if ai.provider in ("openrouter", "nvidia"):
                cards = _generate_with_openrouter(chunk, deck, ai, lang)
            elif ai.provider == "groq":
                cards = _generate_with_groq(chunk, deck, ai, lang)
            else:
                continue

            if cards:
                print(f"  ✅  Fallback succeeded: {ai.name}")
                return cards, ai.name

        except Exception as e:
            err = str(e)
            if "429" in err:
                print(f"  ⏳  {ai.name} rate limited, waiting 30s...")
                time.sleep(30)
            else:
                print(f"  ⚠️  {ai.name} failed: {err[:60]}")
            continue

    return [], "none"


# ── MAIN ORCHESTRATOR ──────────────────────────────────────────────────────────

def generate_from_youtube(
    url:            str,
    deck:           str,
    clients:        list[AIClient],
    detect_topics:  bool = True,
    on_progress:    callable = None,
    lang:           str = "en",
    proxy_config=None,
) -> GenerationResult:
    """
    Full pipeline: YouTube URL → smart chunks → cards.

    Args:
        url:           YouTube URL
        deck:          Anki deck name
        clients:       list of AIClient (sorted by priority internally)
        detect_topics: whether to auto-detect topic per chunk
        on_progress:   optional callback(chunk, total, status_str)
        lang:          output card language — also picks which caption
                       track(s) get requested first (see
                       _transcript_language_priority)
        proxy_config:  optional youtube_transcript_api ProxyConfig — see
                       anki_gen.build_proxy_config(). Works around YouTube
                       blocking cloud-provider IPs (RequestBlocked).
    """
    # Step 1: transcript — prefer a native track in the target language,
    # anki_gen.get_transcript_entries() falls back from there if needed.
    video_id = extract_video_id(url)
    entries  = get_transcript_entries(
        video_id,
        languages=_transcript_language_priority(lang),
        proxy_config=proxy_config,
    )

    # Step 2: smart chunking
    chunks = chunk_youtube(entries)
    print(f"  📦  {len(chunks)} chunks from transcript")

    return _run_pipeline(chunks, deck, clients, detect_topics, on_progress, lang)


def generate_from_pdf(
    path:           str,
    deck:           str,
    clients:        list[AIClient],
    page_from:      int = 1,
    page_to:        int = 99999,
    detect_topics:  bool = True,
    on_progress:    callable = None,
    lang:           str = "en",
) -> GenerationResult:
    """
    Full pipeline: PDF → smart chunks → cards.
    """
    # Step 1: smart chunking
    chunks = chunk_pdf(path, page_from, page_to)
    print(f"  📦  {len(chunks)} chunks from PDF")

    return _run_pipeline(chunks, deck, clients, detect_topics, on_progress, lang)


def _chunk_label(chunk: Chunk) -> str:
    """Where in the source a chunk is: '05:03 – 10:03' (video) or 'pages 4–7' (PDF)."""
    if chunk.source == "youtube":
        return chunk.time_label
    return f"pages {chunk.page_from}–{chunk.page_to}"


def _run_pipeline(
    chunks:         list[Chunk],
    deck:           str,
    clients:        list[AIClient],
    detect_topics:  bool,
    on_progress:    callable,
    lang:           str = "en",
) -> GenerationResult:
    """Internal: run generation pipeline on a list of chunks."""

    # Find primary Gemini client for topic detection
    gemini_clients = [c for c in clients if c.provider == "gemini"]
    primary_gemini = gemini_clients[0] if gemini_clients else None

    result = GenerationResult(
        cards        = [],
        chunks_total = len(chunks),
        chunks_done  = 0,
    )

    all_cards = []

    def _accept(chunk: Chunk, cards: list[dict], provider: str):
        """Tag the cards with the chunk topic and count the chunk as done."""
        for card in cards:
            if chunk.topic_hint:
                safe_tag = re.sub(r"\s+", "_", chunk.topic_hint.lower())[:30]
                card.setdefault("tags", [])
                if safe_tag not in card["tags"]:
                    card["tags"].append(safe_tag)
        all_cards.extend(cards)
        result.chunks_done += 1
        result.ai_used[provider] = result.ai_used.get(provider, 0) + 1

    # ── Pass 1: every chunk ───────────────────────────────────────────────────
    pending = []                                  # chunks that produced nothing

    for chunk in chunks:

        # Step 3: detect topic
        if detect_topics and primary_gemini:
            chunk.topic_hint = detect_topic(chunk, primary_gemini)
            print(f"  🏷️   Chunk {chunk.index}/{chunk.total}: {chunk.topic_hint}")
        else:
            chunk.topic_hint = _fallback_topic(chunk)

        # Step 4: generate cards
        print(f"  🔄  Generating chunk {chunk.index}/{chunk.total}...", end=" ", flush=True)
        cards, provider = generate_chunk(chunk, deck, clients, lang)
        print(f"{len(cards)} cards [{provider}]")

        if cards:
            _accept(chunk, cards, provider)
        else:
            pending.append(chunk)

        # Progress callback for web UI
        if on_progress:
            on_progress(chunk.index, chunk.total, chunk.topic_hint)

    # ── Pass 2: one more chance for the chunks that failed ────────────────────
    # Most failures are rate limits that clear within a minute. Silently dropping
    # those parts of the source would leave holes in the deck nobody notices.
    if pending:
        print(f"\n  🔁  {len(pending)} chunk(s) failed — retrying once after {SECOND_PASS_COOLDOWN}s...")
        time.sleep(SECOND_PASS_COOLDOWN)
        still_failing = []
        for chunk in pending:
            print(f"  🔄  Retrying chunk {chunk.index}/{chunk.total}...", end=" ", flush=True)
            cards, provider = generate_chunk(chunk, deck, clients, lang)
            print(f"{len(cards)} cards [{provider}]")
            if cards:
                _accept(chunk, cards, provider)
            else:
                still_failing.append(chunk)
        pending = still_failing

    for chunk in pending:
        result.failed_chunks.append(chunk.index)
        result.errors.append(f"Chunk {chunk.index} ({_chunk_label(chunk)}): no cards generated")

    # Exact deduplication (semantic dedup handled by filter.py)
    result.cards = deduplicate(all_cards)

    # Summary
    print(f"\n  ✅  {len(result.cards)} unique cards from {result.chunks_done}/{result.chunks_total} chunks")
    print(f"  🤖  AI usage: {result.ai_used}")

    return result