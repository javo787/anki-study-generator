"""
quality.py — Final quality control using Gemini Pro
Adds mnemonics, improves phrasing, verifies medical accuracy.
"""

import re
import sys
import json
import time
from dataclasses import dataclass, field

try:
    from google import genai
except ImportError:
    sys.exit("❌  Run: pip install google-genai")

try:
    from anki_gen import build_language_instruction
except ImportError:
    def build_language_instruction(lang: str) -> str:   # pragma: no cover — fallback if run standalone
        return "Write every \"front\" and \"back\" field in English."

GEMINI_PRO_MODEL   = "gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
QUALITY_BATCH      = 20
MAX_RETRY          = 3
RETRY_WAIT         = 15


@dataclass
class QualityResult:
    cards_in:          int = 0
    cards_out:         int = 0
    mnemonics_added:   int = 0
    phrasing_improved: int = 0
    errors:            list = field(default_factory=list)
    model_used:        str  = ""

    def summary(self) -> str:
        return (
            f"Quality: {self.cards_in} in → {self.cards_out} out | "
            f"mnemonics added: {self.mnemonics_added} | "
            f"phrasing improved: {self.phrasing_improved} | "
            f"model: {self.model_used}"
        )


QUALITY_SYSTEM = """\
You are a senior medical educator reviewing Anki flashcards for final quality.

Your tasks for EACH card:
1. MNEMONIC — add a 💡 mnemonic if the card has a list of 3+ items AND no mnemonic exists yet
2. PHRASING — improve vague questions
3. ACCURACY — fix any medical inaccuracies silently
4. CLOZE FORMAT — for "type": "cloze" cards, "front" MUST use real Anki
   syntax: "What is the outermost meningeal layer?<br><br>{{c1::Dura mater}}".
   NEVER use "___" placeholders — Anki generates zero cards for a cloze
   note with no {{cN::...}} markup, so a "___" card is silently dead.
5. FORMATTING — HTML only, never markdown (no **bold**, use <b>bold</b>
   instead) — the output is rendered as raw HTML.
6. LANGUAGE — {language_instruction} This applies to any text you add or
   rewrite too: a new mnemonic must be written in that same language/script,
   not English by default.

IMPORTANT:
- Do NOT remove cards
- Do NOT change card type
- Do NOT add mnemonics to every card — only where genuinely useful
- Preserve all existing tags

OUTPUT: JSON only — no markdown.
{
  "cards": [
    {
      "front": "...",
      "back": "...",
      "type": "basic",
      "tags": ["..."],
      "improved": true,
      "mnemonic_added": false
    }
  ]
}"""


def _process_batch(cards, deck, client, model, lang="en"):
    prompt = (
        f"Review and improve these {len(cards)} Anki cards.\n"
        f"Deck: {deck}\n\n"
        f"Cards:\n{json.dumps(cards, indent=2, ensure_ascii=False)}\n\n"
        f"Return improved cards as JSON only."
    )
    system_instruction = QUALITY_SYSTEM.format(
        language_instruction=build_language_instruction(lang)
    )

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.models.generate_content(
                model    = model,
                contents = prompt,
                config   = {
                    "system_instruction": system_instruction,
                    "temperature":        0.4,
                },
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data           = json.loads(raw)
            improved_cards = data.get("cards", cards)

            mnemonics = sum(1 for c in improved_cards if c.get("mnemonic_added", False))
            phrasing  = sum(1 for c in improved_cards if c.get("improved", False) and not c.get("mnemonic_added", False))

            for card in improved_cards:
                card.pop("improved",       None)
                card.pop("mnemonic_added", None)

            return improved_cards, mnemonics, phrasing

        except json.JSONDecodeError as e:
            print(f"  ⚠️  Quality JSON parse error: {e}")
            return cards, 0, 0

        except Exception as e:
            err = str(e)
            if ("503" in err or "overload" in err.lower()) and attempt < MAX_RETRY:
                wait = attempt * RETRY_WAIT
                print(f"  ⏳  Gemini overloaded, retry {attempt}/{MAX_RETRY} in {wait}s...")
                time.sleep(wait)
            elif "429" in err and attempt < MAX_RETRY:
                wait = attempt * 30
                print(f"  ⏳  Rate limit, retry {attempt}/{MAX_RETRY} in {wait}s...")
                time.sleep(wait)
            elif "pro" in model.lower() and attempt == MAX_RETRY:
                raise RuntimeError(f"Pro unavailable: {err}")
            else:
                print(f"  ⚠️  Gemini quality error: {err[:80]}")
                return cards, 0, 0

    return cards, 0, 0


def improve_cards(
    cards:   list,
    deck:    str,
    api_key: str  = "",
    use_pro: bool = True,
    lang:    str  = "en",
) -> tuple:
    if not cards:
        return [], QualityResult(cards_in=0, cards_out=0)

    if not api_key:
        return cards, QualityResult(
            cards_in  = len(cards),
            cards_out = len(cards),
            model_used = "skipped"
        )

    client     = genai.Client(api_key=api_key)
    model      = GEMINI_PRO_MODEL if use_pro else GEMINI_FLASH_MODEL
    cards_in   = len(cards)
    all_improved     = []
    total_mnemonics  = 0
    total_phrasing   = 0
    errors           = []
    model_used       = model

    batches = [cards[i:i + QUALITY_BATCH] for i in range(0, len(cards), QUALITY_BATCH)]
    print(f"\n  ✨  Quality check: {cards_in} cards in {len(batches)} batch(es) [{model}]...")

    # цикл по батчам перед отправкой в Gemini (пример вставки)
    for i, batch in enumerate(batches, 1):
        # Нормализуй перед отправкой в Gemini
        for card in batch:
            if isinstance(card.get("back"), list):
                card["back"] = "\n".join(str(x) for x in card["back"])
            if isinstance(card.get("front"), list):
                card["front"] = " ".join(str(x) for x in card["front"])

        print(f"  🔄  Quality batch {i}/{len(batches)}...", end=" ", flush=True)
        try:
            improved, mnemonics, phrasing = _process_batch(batch, deck, client, model, lang)
            model_used = model
        except RuntimeError as e:
            if use_pro:
                print(f"\n  ⚠️  Pro unavailable, falling back to Flash...")
                model      = GEMINI_FLASH_MODEL
                model_used = model
                try:
                    improved, mnemonics, phrasing = _process_batch(batch, deck, client, model, lang)
                except Exception as e2:
                    print(f"skipped ({e2.__class__.__name__})")
                    improved, mnemonics, phrasing = batch, 0, 0
                    errors.append(f"Batch {i}: {str(e2)[:60]}")
            else:
                improved, mnemonics, phrasing = batch, 0, 0
                errors.append(f"Batch {i}: {str(e)[:60]}")

        print(f"done (+{mnemonics} mnemonics, +{phrasing} improved)")
        all_improved    += improved
        total_mnemonics += mnemonics
        total_phrasing  += phrasing

        if i < len(batches):
            time.sleep(3)

    result = QualityResult(
        cards_in          = cards_in,
        cards_out         = len(all_improved),
        mnemonics_added   = total_mnemonics,
        phrasing_improved = total_phrasing,
        errors            = errors,
        model_used        = model_used,
    )
    print(f"\n  {result.summary()}")
    return all_improved, result


_SPECIFIC_QUESTION_WORDS = {
    "en": ["which", "where", "name", "list", "what are"],
    "ru": ["какой", "какая", "какие", "где", "перечисли", "назовите", "сколько"],
    "de": ["welche", "welcher", "wo", "nenne", "nennen sie", "wie viele"],
    "uz": ["qaysi", "қайси", "qayerda", "қаерда", "sanab", "санаб", "nechta", "нечта"],
}


def score_cards(cards: list, lang: str = "en") -> list:
    question_words = _SPECIFIC_QUESTION_WORDS.get((lang or "en").lower(), _SPECIFIC_QUESTION_WORDS["en"])

    def _score(card: dict) -> int:
        score = 0
        front = card.get("front", "") or ""
        back  = card.get("back",  "") or ""
        
        # Если back пришёл списком — конвертируй в строку
        if isinstance(back, list):
            back = " ".join(str(x) for x in back)
        if isinstance(front, list):
            front = " ".join(str(x) for x in front)
        
        if any(w in front.lower() for w in question_words):
            score += 2
        if "💡" in back:
            score += 3
        if "•" in back:
            score += 1
        if re.search(r"\d+", back):
            score += 1
        if card.get("type") == "cloze":
            score += 1
        if len(back) < 20:
            score -= 2
        return score

    return sorted(cards, key=_score, reverse=True)