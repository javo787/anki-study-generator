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
    ai_used:      dict      = field(default_factory=dict)  # provider → count

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
5. Cloze format: "The ___ is the outermost meninges layer. (Dura mater)"
6. Minimum 15, maximum 35 cards
7. Topic: {topic} | Chunk {chunk_index} of {chunk_total}

OUTPUT: JSON only — no markdown, no explanation.
{{"cards": [{{"front": "...", "back": "...", "type": "basic", "tags": ["..."]}}]}}"""

def _parse_response(raw: str) -> list[dict]:
    """Strip markdown fences and parse JSON — handles all Gemini response formats."""
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    # If response starts with { directly
    if raw.startswith("{"):
        return json.loads(raw).get("cards", [])

    # Try to find JSON object anywhere in response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group()).get("cards", [])

    return []

def _generate_with_groq(chunk: Chunk, deck: str, ai: AIClient) -> list[dict]:
    """Generate cards using Groq API."""
    system = GROQ_CARD_PROMPT.format(
        topic=chunk.topic_hint or "general",
        chunk_index=chunk.index,
        chunk_total=chunk.total,
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


def _generate_with_openrouter(chunk: Chunk, deck: str, ai: AIClient) -> list[dict]:
    """Generate cards using OpenRouter API (OpenAI-compatible)."""
    system = GROQ_CARD_PROMPT.format(
        topic=chunk.topic_hint or "general",
        chunk_index=chunk.index,
        chunk_total=chunk.total,
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

                elif "429" in err:
                    print(f"  ⏳  Gemini 429 rate limit — switching to fallback immediately...")
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
                cards = _generate_with_openrouter(chunk, deck, ai)
            elif ai.provider == "groq":
                cards = _generate_with_groq(chunk, deck, ai)
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
) -> GenerationResult:
    """
    Full pipeline: YouTube URL → smart chunks → cards.

    Args:
        url:           YouTube URL
        deck:          Anki deck name
        clients:       list of AIClient (sorted by priority internally)
        detect_topics: whether to auto-detect topic per chunk
        on_progress:   optional callback(chunk, total, status_str)
    """
    # Step 1: transcript
    video_id = extract_video_id(url)
    entries  = get_transcript_entries(video_id)

    # Step 2: smart chunking
    chunks = chunk_youtube(entries)
    print(f"  📦  {len(chunks)} chunks from transcript")

    return _run_pipeline(chunks, deck, clients, detect_topics, on_progress)


def generate_from_pdf(
    path:           str,
    deck:           str,
    clients:        list[AIClient],
    page_from:      int = 1,
    page_to:        int = 99999,
    detect_topics:  bool = True,
    on_progress:    callable = None,
) -> GenerationResult:
    """
    Full pipeline: PDF → smart chunks → cards.
    """
    # Step 1: smart chunking
    chunks = chunk_pdf(path, page_from, page_to)
    print(f"  📦  {len(chunks)} chunks from PDF")

    return _run_pipeline(chunks, deck, clients, detect_topics, on_progress)


def _run_pipeline(
    chunks:         list[Chunk],
    deck:           str,
    clients:        list[AIClient],
    detect_topics:  bool,
    on_progress:    callable,
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

    for chunk in chunks:

        # Step 3: detect topic
        if detect_topics and primary_gemini:
            chunk.topic_hint = detect_topic(chunk, primary_gemini)
            print(f"  🏷️   Chunk {chunk.index}/{chunk.total}: {chunk.topic_hint}")
        else:
            chunk.topic_hint = _fallback_topic(chunk)

        # Step 4: generate cards
        print(f"  🔄  Generating chunk {chunk.index}/{chunk.total}...", end=" ", flush=True)
        cards, provider = generate_chunk(chunk, deck, clients)
        print(f"{len(cards)} cards [{provider}]")

        # Track which AI was used
        result.ai_used[provider] = result.ai_used.get(provider, 0) + 1

        if cards:
            # Tag cards with topic
            for card in cards:
                if chunk.topic_hint:
                    safe_tag = re.sub(r"\s+", "_", chunk.topic_hint.lower())[:30]
                    card.setdefault("tags", [])
                    if safe_tag not in card["tags"]:
                        card["tags"].append(safe_tag)

            all_cards.extend(cards)
            result.chunks_done += 1
        else:
            result.errors.append(f"Chunk {chunk.index}: no cards generated")

        # Progress callback for web UI
        if on_progress:
            on_progress(chunk.index, chunk.total, chunk.topic_hint)

    # Exact deduplication (semantic dedup handled by filter.py)
    result.cards = deduplicate(all_cards)

    # Summary
    print(f"\n  ✅  {len(result.cards)} unique cards from {result.chunks_done}/{result.chunks_total} chunks")
    print(f"  🤖  AI usage: {result.ai_used}")

    return result