"""
pdf_gen.py — PDF Card Generation Source
Requires: pip install google-genai pdfplumber
"""

import re
import sys
import json
import os
import time

# ── IMPORTS ────────────────────────────────────────────────────────────────────

try:
    import pdfplumber
except ImportError:
    sys.exit("❌  Run: pip install pdfplumber")

try:
    from google import genai
except ImportError:
    sys.exit("❌  Run: pip install google-genai")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

MODEL      = "gemini-2.5-flash"
MAX_RETRY  = 3
RETRY_WAIT = 10

MEDICAL_PROMPT = """\
You are an expert medical Anki card creator trained on First Aid, Najeeb lectures, and Mnemosyne decks.

RULES:
1. One card = one testable fact (anatomy, function, pathway, number, clinical correlation)
2. Front: specific question — What / Where / Which / How / Name / List
3. Back: clean bullet list (max 5 bullets) + 💡 mnemonic or clinical note where genuinely useful
4. Avoid vague questions like "What is X?" — prefer "What are the branches of X?"
5. Generate both Basic Q&A and Cloze deletion cards
6. Cloze format: "The ___ supplies the SA node. (Right coronary artery)"
7. Skip page numbers, headings, figure captions, filler text
8. You are processing chunk {chunk_index} of {chunk_total} — do NOT repeat facts from previous chunks
9. Topic of this chunk: {topic}
10. Minimum 15 cards, maximum 35 cards per chunk

OUTPUT: JSON only — no markdown, no explanation.

{{
  "cards": [
    {{
      "front": "What are the branches of the celiac trunk?",
      "back": "• Left gastric artery\\n• Splenic artery\\n• Common hepatic artery\\n💡 Let's Grab Chicken (LGC)",
      "type": "basic",
      "tags": ["anatomy", "vasculature", "celiac"]
    }},
    {{
      "front": "The ___ is the most commonly occluded coronary artery. (LAD)",
      "back": "LAD (Left Anterior Descending)\\n💡 LAD = Widow Maker",
      "type": "cloze",
      "tags": ["cardiology", "coronary"]
    }}
  ]
}}"""

LANGUAGE_PROMPT = """\
You are an expert German language Anki card creator for medical students using Mensch textbook.

RULES:
1. Extract: nouns (with article + plural), verbs (infinitive + Perfekt), adjectives, useful phrases
2. Basic card — Front: German word/phrase, Back: English + Russian translation + example sentence
3. Cloze card — take example sentence, blank the key word: "Ich ___ ins Kino gegangen. (bin)"
4. For nouns always include: der/die/das + plural form on the back
5. For verbs include: infinitive + Perfekt form
6. Prioritize vocabulary that appears multiple times or is medically relevant
7. You are processing chunk {chunk_index} of {chunk_total} — do NOT repeat vocabulary from previous chunks
8. Topic of this chunk: {topic}
9. Minimum 10, maximum 25 cards per chunk

OUTPUT: JSON only — no markdown, no explanation.

{{
  "cards": [
    {{
      "front": "der Arzt / die Ärztin",
      "back": "doctor (EN) / врач (RU)\\nPlural: die Ärzte\\nExample: Der Arzt untersucht den Patienten.",
      "type": "basic",
      "tags": ["vocabulary", "nouns", "professions"]
    }},
    {{
      "front": "Der ___ untersucht den Patienten. (Arzt)",
      "back": "Arzt\\n💡 der Arzt = male | die Ärztin = female",
      "type": "cloze",
      "tags": ["cloze", "vocabulary"]
    }}
  ]
}}"""

# ── HELPERS ────────────────────────────────────────────────────────────────────

def detect_mode(deck: str) -> str:
    """Detect card mode from deck name."""
    lower = deck.lower()
    if any(w in lower for w in ["deutsch", "german", "mensch", "sprache", "vocabulary", "vokabel"]):
        return "language"
    return "medical"


def extract_pdf_text(path: str, page_from: int = 1, page_to: int = 99999) -> tuple[str, int]:
    """
    Extract text from PDF between given page numbers (1-indexed).
    Returns (text, total_pages).
    Note: chunking is handled by chunker.py — this just extracts raw text.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    texts       = []
    total_pages = 0

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        page_from   = max(1, page_from)
        page_to     = min(total_pages, page_to)

        for i in range(page_from - 1, page_to):
            page_text = pdf.pages[i].extract_text()
            if page_text and page_text.strip():
                texts.append(f"[Page {i + 1}]\n{page_text.strip()}")

    if not texts:
        raise RuntimeError(
            "No text extracted. PDF might be scanned/image-based. "
            "Try OCR software first."
        )

    return "\n\n".join(texts), total_pages


def _parse_response(raw: str) -> list[dict]:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw).get("cards", [])


def generate_cards(
    chunk:       str,
    mode:        str,
    deck:        str,
    lang:        str,
    client,
    chunk_index: int = 1,
    chunk_total: int = 1,
    topic:       str = "general",
) -> list[dict]:
    """
    Generate Anki cards from a PDF text chunk using Gemini.

    Args:
        chunk:       text content to process
        mode:        "medical" | "language"
        deck:        deck name for context
        lang:        "en" | "ru" | "de"
        client:      Gemini client instance
        chunk_index: current chunk number
        chunk_total: total chunks
        topic:       topic hint from pipeline

    Returns:
        list of card dicts with keys: front, back, type, tags
    """
    system_template = MEDICAL_PROMPT if mode == "medical" else LANGUAGE_PROMPT
    system = system_template.format(
        chunk_index = chunk_index,
        chunk_total = chunk_total,
        topic       = topic,
    )

    lang_note = {
        "ru": "\nWrite all card fronts and backs in Russian.",
        "en": "\nWrite all card fronts and backs in English.",
        "de": "\nWrite German on the front, English + Russian on the back.",
    }.get(lang, "")

    prompt = (
        f"Deck: {deck}\n"
        f"Mode: {mode}\n"
        f"Chunk: {chunk_index} of {chunk_total}\n"
        f"Topic: {topic}\n\n"
        f"Textbook excerpt:\n{chunk}\n\n"
        f"Generate Anki flashcards. JSON only."
    )

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.models.generate_content(
                model   = MODEL,
                contents = prompt,
                config  = {
                    "system_instruction": system + lang_note,
                    "temperature":        0.3,
                },
            )
            return _parse_response(response.text)

        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error (chunk {chunk_index}): {e}")
            return []

        except Exception as e:
            err = str(e)
            if "503" in err and attempt < MAX_RETRY:
                wait = attempt * RETRY_WAIT
                print(f"  ⏳  503 overload, retry {attempt}/{MAX_RETRY} in {wait}s...")
                time.sleep(wait)
            elif "429" in err and attempt < MAX_RETRY:
                wait = attempt * 30
                print(f"  ⏳  429 rate limit, retry {attempt}/{MAX_RETRY} in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ⚠️  Gemini error (chunk {chunk_index}): {e}")
                return []

    return []


def deduplicate(cards: list[dict]) -> list[dict]:
    """Remove exact duplicate cards by front text."""
    seen, unique = set(), []
    for c in cards:
        k = c["front"].lower().strip()
        if k not in seen:
            seen.add(k)
            unique.append(c)
    return unique


def build_client(api_key: str):
    """Build and return a Gemini client."""
    return genai.Client(api_key=api_key)