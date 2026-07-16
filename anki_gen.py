"""
Anki Card Generator — YouTube Source
Requires: pip install google-genai youtube-transcript-api
"""

import re
import sys
import json
import time

# ── IMPORTS ────────────────────────────────────────────────────────────────────

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    sys.exit("❌  Run: pip install youtube-transcript-api")

try:
    from google import genai
except ImportError:
    sys.exit("❌  Run: pip install google-genai")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

MODEL      = "gemini-2.5-flash"
MAX_RETRY  = 3
RETRY_WAIT = 10  # seconds, multiplied by attempt number

SYSTEM_PROMPT = """\
You are an expert medical Anki card creator trained on First Aid, Anki Mnemosyne decks, and Najeeb lectures.

CARD QUALITY RULES:
1. One card = one testable fact (anatomy, function, pathway, number, clinical correlation)
2. Front: specific question — What / Where / Which / How / Name / List
3. Back: clean bullet list (max 5 bullets) + 💡 mnemonic or key clinical note where genuinely useful
4. Avoid vague questions like "What is X?" — prefer "What are the branches of X?" or "Where does X drain?"
5. Skip filler, greetings, meta-commentary from the transcript
6. You are processing chunk {chunk_index} of {chunk_total} — do NOT repeat cards from previous chunks
7. Topic of this chunk: {topic}
8. Minimum 15 cards, maximum 35 cards per chunk
9. FORMATTING: use HTML only, never markdown (no **bold**, use <b>bold</b>)
10. For "type": "cloze" cards, "front" MUST contain real Anki cloze syntax
    {{{{c1::answer}}}} — NEVER "___" placeholders. Anki only creates a card
    for each {{{{cN::...}}}} it finds in the text; without it, no card is made at all.

OUTPUT: JSON only — no markdown fences, no explanation, no preamble.

{{
  "cards": [
    {{
      "front": "What are the three layers of the meninges from outside to inside?",
      "back": "• Dura mater\\n• Arachnoid mater\\n• Pia mater\\n💡 Mnemonic: <b>DAP</b> — Dura, Arachnoid, Pia",
      "type": "basic",
      "tags": ["neuroanatomy", "meninges"]
    }},
    {{
      "front": "What is the outermost layer of the meninges?<br><br>{{{{c1::Dura mater}}}}",
      "back": "💡 Dura = hard in Latin — it is the toughest layer",
      "type": "cloze",
      "tags": ["neuroanatomy", "meninges"]
    }}
  ]
}}"""

# ── TRANSCRIPT ─────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any URL format."""
    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"v=([A-Za-z0-9_-]{11})",
        r"embed/([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    raise ValueError(f"Cannot extract video ID from: {url}")


def get_transcript_entries(video_id: str) -> list[dict]:
    """
    Return raw transcript entries with timestamps.
    Each entry: {"text": str, "start": float, "duration": float}
    Used by chunker.py for smart splitting.
    """
    try:
        fetcher = YouTubeTranscriptApi()
        entries = fetcher.fetch(video_id)
        return [
            {
                "text":     e.text.strip(),
                "start":    e.start,
                "duration": e.duration,
            }
            for e in entries
            if e.text.strip()
        ]
    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {e}")


def get_transcript(video_id: str) -> str:
    """Return plain text transcript. Backwards compatible."""
    entries = get_transcript_entries(video_id)
    return " ".join(e["text"] for e in entries)


# ── CARD GENERATION ────────────────────────────────────────────────────────────

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


def generate_cards(
    chunk:       str,
    deck:        str,
    client,
    chunk_index: int = 1,
    chunk_total: int = 1,
    topic:       str = "general",
) -> list[dict]:
    """
    Generate Anki cards from a text chunk using Gemini.

    Args:
        chunk:       text content to process
        deck:        deck name (used as context)
        client:      Gemini client instance
        chunk_index: current chunk number (for context)
        chunk_total: total chunks (for context)
        topic:       topic hint from pipeline (improves quality)

    Returns:
        list of card dicts with keys: front, back, type, tags
    """
    system = SYSTEM_PROMPT.format(
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        topic=topic,
    )

    prompt = (
        f"Deck: {deck}\n"
        f"Chunk: {chunk_index} of {chunk_total}\n"
        f"Topic: {topic}\n\n"
        f"Transcript:\n{chunk}\n\n"
        f"Generate Anki cards. JSON only."
    )

    # заменённый одиночный вызов без внутреннего retry
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "system_instruction": system,
                "temperature": 0.3,
            },
        )
        return _parse_response(response.text)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error (chunk {chunk_index}): {e}")
        return []
    except Exception:
        raise  # пробрасываем исключение наверх в generate_chunk

    return []


# ── DEDUPLICATION ──────────────────────────────────────────────────────────────

def deduplicate(cards: list[dict]) -> list[dict]:
    """
    Remove exact duplicate cards by front text.
    Semantic deduplication is handled later by filter.py (Groq).
    """
    seen, unique = set(), []
    for c in cards:
        k = c["front"].lower().strip()
        if k not in seen:
            seen.add(k)
            unique.append(c)
    return unique


# ── CLIENT ─────────────────────────────────────────────────────────────────────

def build_client(api_key: str):
    """Build and return a Gemini client."""
    return genai.Client(api_key=api_key)