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

# ── LANGUAGE ───────────────────────────────────────────────────────────────
# Shared by anki_gen.py, generator.py (fallback providers) and quality.py, so
# every prompt that writes or edits card text agrees on what "Card language"
# actually means. Previously `lang` was accepted by the pipeline's public
# functions but never reached any prompt — the picker in the UI had no
# effect at all on YouTube/PDF generation. This is what actually wires it up.
def build_language_instruction(lang: str) -> str:
    lang = (lang or "en").lower()
    if lang == "uz":
        return (
            'Write every "front" and "back" field in Uzbek, using the '
            "CYRILLIC alphabet (Ўзбек кирилл ёзуви) — NEVER the Latin/Roman "
            'alphabet. For example write "бош мия" and "юрак", never '
            '"bosh miya" or "yurak". Use standard Uzbek medical '
            "terminology; well-known international abbreviations (ДНК, МРТ, "
            "АТФ etc.) may stay in the form commonly used in Uzbek medical "
            "texts."
        )
    if lang == "ru":
        return 'Write every "front" and "back" field in Russian, using standard Russian medical terminology.'
    if lang == "de":
        return 'Write every "front" and "back" field in German (Deutsch).'
    return 'Write every "front" and "back" field in English.'


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
11. LANGUAGE: {language_instruction}

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


def get_transcript_entries(video_id: str, languages: list[str] | None = None) -> list[dict]:
    """
    Return raw transcript entries with timestamps.
    Each entry: {"text": str, "start": float, "duration": float}
    Used by chunker.py for smart splitting.

    languages: priority list of language codes to look for, e.g. ["uz", "en"].
    Defaults to ["en"] — matches the old hardcoded behaviour exactly, so any
    existing caller that doesn't pass `languages` is unaffected.

    Root bug this fixes: this used to call fetcher.fetch(video_id) with no
    `languages` arg at all, which youtube-transcript-api silently defaults
    to ('en',). For any non-English video without an English caption track
    (i.e. most Russian/Uzbek/German lecture videos), that raised inside
    generate_from_youtube() regardless of what "Card language" was selected
    in the UI — the language picker never actually reached this call.

    Fetch strategy (each step only runs if the previous one fails):
      1. Fetch a transcript natively in one of `languages` (manually created
         captions are preferred over auto-generated ones by the library).
      2. No native transcript in any of `languages` — take whatever
         transcript IS available on the video and, if YouTube allows
         translating it, translate it into the first language requested.
      3. Translation isn't available either — fall back to whatever
         transcript exists, untranslated (still better than failing the
         whole pipeline; the generation step's own language instruction
         will still produce output in the requested language/script).
    """
    languages = list(languages) if languages else ["en"]
    fetcher   = YouTubeTranscriptApi()

    def _entries(fetched) -> list[dict]:
        return [
            {"text": e.text.strip(), "start": e.start, "duration": e.duration}
            for e in fetched
            if e.text.strip()
        ]

    # ── 1. Direct match ────────────────────────────────────────────────────
    try:
        fetched = fetcher.fetch(video_id, languages=languages)
        print(f"  🌐  Transcript: native '{fetched.language_code}' track")
        return _entries(fetched)
    except Exception:
        pass

    # ── 2/3. Fall back to whatever exists, translating if possible ────────
    try:
        transcript_list = fetcher.list(video_id)
        available = transcript_list.find_transcript(
            [t.language_code for t in transcript_list]
        )
        wanted            = languages[0]
        translation_codes = {t.language_code for t in available.translation_languages}

        if available.language_code != wanted and available.is_translatable and wanted in translation_codes:
            fetched = available.translate(wanted).fetch()
            print(f"  🌐  Transcript: translated '{available.language_code}' → '{wanted}'")
        else:
            fetched = available.fetch()
            print(f"  🌐  Transcript: no '{'/'.join(languages)}' track — "
                  f"using '{available.language_code}' instead")
        return _entries(fetched)

    except Exception as e:
        raise RuntimeError(
            f"Could not fetch a transcript for this video in {languages}: {e}. "
            f"If this video truly has no captions in these languages, download "
            f"it and use the 'Local video' source instead — Whisper transcribes "
            f"straight from audio, so caption availability doesn't matter there."
        )


def get_transcript(video_id: str, languages: list[str] | None = None) -> str:
    """Return plain text transcript. Backwards compatible."""
    entries = get_transcript_entries(video_id, languages=languages)
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
    lang:        str = "en",
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
        lang:        output card language — "en" | "ru" | "de" | "uz"
                     ("uz" writes Cyrillic script, see build_language_instruction)

    Returns:
        list of card dicts with keys: front, back, type, tags
    """
    system = SYSTEM_PROMPT.format(
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        topic=topic,
        language_instruction=build_language_instruction(lang),
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