"""
format_pass.py — Dedicated card FORMAT pass (runs after quality.py, before save).

Why this exists as its own step instead of being folded into quality.py:
Formatting (choosing real Anki cloze syntax, HTML vs markdown, hint sub-fields,
grouping related facts under one cloze number) is a distinct skill from
"does this card teach the right thing" (quality.py's job) or "is this card
worth keeping" (filter.py's job). Mixing the instructions weakens all three.

Root bug this fixes: cards of type "cloze" were being generated as
"The ___ drains CSF. (Arachnoid villi)" — plain text with NO real Anki
cloze markup ({{c1::...}}). Anki's Cloze notetype only generates a card
for each {{cN::...}} found in the Text field — with none present, the
note silently produces NO card at all. This pass makes the model emit
real {{c1::answer}} / {{c1::answer::hint}} syntax directly, and validates
it before the card ever reaches save_cards().
"""

import re
import sys
import json
import time
from dataclasses import dataclass, field

try:
    from groq import Groq
except ImportError:
    sys.exit("❌  Run: pip install groq")

from uz_transliterate import to_cyrillic, has_cyrillic

GROQ_MODEL   = "llama-3.3-70b-versatile"
FORMAT_BATCH = 25
MAX_RETRY    = 3
RETRY_WAIT   = 10


@dataclass
class FormatResult:
    cards_in:          int  = 0
    cards_out:         int  = 0
    cloze_fixed:       int  = 0   # cards that had broken/missing {{c1::}} syntax, repaired
    cloze_demoted:     int  = 0   # cloze cards that couldn't be fixed, safely demoted to basic
    markdown_stripped:  int = 0
    uz_transliterated: int  = 0   # fields converted Latin -> Cyrillic Uzbek (lang="uz" only)
    errors:            list = field(default_factory=list)

    def summary(self) -> str:
        extra = f" | uz→Cyrillic: {self.uz_transliterated}" if self.uz_transliterated else ""
        return (
            f"Format: {self.cards_in} in → {self.cards_out} out | "
            f"cloze fixed: {self.cloze_fixed} | "
            f"cloze demoted (unfixable): {self.cloze_demoted} | "
            f"markdown→HTML: {self.markdown_stripped}{extra}"
        )


FORMAT_SYSTEM = """\
You are a senior Anki flashcard FORMATTER, trained on professional medical \
decks (AnKing, Zanki, First-Aid-based Mnemosyne collections).

Your ONLY job is to fix FORMAT. Do not change what each card teaches, do \
not add or remove cards, do not change tags.

RULES:

1. HTML ONLY — never use markdown **bold** or _italic_. The output is \
rendered as raw HTML (#html:true), so markdown symbols show up as literal \
asterisks on the card. Use <b>...</b> for emphasis instead.

2. REAL ANKI CLOZE SYNTAX — for any card with "type": "cloze", the "front" \
field MUST contain real Anki cloze markup: {{c1::answer}}. NEVER output \
"___" placeholders or "(answer)" written in parentheses — those produce \
NO card at all when imported into Anki, because Anki only creates a card \
for each {{cN::...}} it finds in the text.
   - Preferred structure: "<question>?<br><br>{{c1::answer}}"
     e.g. "What produces CSF?<br><br>{{c1::Choroid plexus}}"
   - Or inline in a declarative sentence:
     "The {{c1::dura mater}} is the outermost meningeal layer."
   - If the blanked term alone is ambiguous (a number, a direction, a \
side, increased/decreased), add a hint after a second "::":
     {{c1::increased::increased/decreased}}, {{c1::2::number of layers}}
   - If 2+ facts in the same card should be recalled TOGETHER as one \
event, reuse the same number (c1) for both. If they are independent facts \
that deserve separate review cards, use different numbers (c1, c2, ...).

3. Keep every other field (back, tags) as close to the original as \
possible — only touch formatting, not content or difficulty.

OUTPUT: JSON only — no markdown fences, no explanation.
{
  "cards": [
    {
      "front": "What produces cerebrospinal fluid?<br><br>{{c1::Choroid plexus}}",
      "back": "\\ud83d\\udca1 Located within the ventricles",
      "type": "cloze",
      "tags": ["neuroanatomy", "CSF"]
    },
    {
      "front": "What are the three meningeal layers from outside to inside?",
      "back": "\\u2022 <b>D</b>ura mater<br>\\u2022 <b>A</b>rachnoid mater<br>\\u2022 <b>P</b>ia mater<br>\\ud83d\\udca1 Mnemonic: <b>DAP</b>",
      "type": "basic",
      "tags": ["neuroanatomy", "meninges"]
    }
  ]
}"""

FORMAT_USER = """\
Fix the FORMAT of these {count} Anki cards. Deck: {deck}

Cards:
{cards_json}

Return JSON only — same number of cards, same content, fixed format."""


def _has_real_cloze(text: str) -> bool:
    return bool(re.search(r"\{\{c\d+::", text or ""))


def _convert_legacy_cloze(text: str) -> str | None:
    """
    Best-effort local fallback for a "___...(answer)" style card that the
    AI failed to convert. Returns None if it can't be safely converted.
    """
    match = re.search(r"\(([^)]+)\)\s*$", text or "")
    if not match:
        return None
    answer   = match.group(1).strip()
    question = text[:match.start()].strip()
    if "___" not in question:
        return None
    return question.replace("___", f"{{{{c1::{answer}}}}}")


def _strip_markdown_bold(text: str) -> tuple[str, bool]:
    """Fallback safety net: convert any remaining **bold** to <b>bold</b>."""
    if not text or "**" not in text:
        return text, False
    fixed = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return fixed, fixed != text


def _format_batch(cards: list[dict], deck: str, client: Groq) -> list[dict]:
    cards_json = json.dumps(cards, indent=2, ensure_ascii=False)
    prompt = FORMAT_USER.format(count=len(cards), deck=deck, cards_json=cards_json)

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model    = GROQ_MODEL,
                messages = [
                    {"role": "system", "content": FORMAT_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature = 0.1,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            return data.get("cards", cards)

        except json.JSONDecodeError as e:
            print(f"  ⚠️  Format JSON parse error: {e}")
            return cards

        except Exception as e:
            err = str(e)
            if ("503" in err or "overload" in err.lower() or "429" in err) and attempt < MAX_RETRY:
                wait = attempt * RETRY_WAIT
                print(f"  ⏳  Groq busy, retry {attempt}/{MAX_RETRY} in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ⚠️  Format step error: {err[:80]}")
                return cards

    return cards


def format_cards(
    cards:        list[dict],
    deck:         str,
    groq_api_key: str = "",
    lang:         str = "en",
) -> tuple[list[dict], FormatResult]:
    """
    Run the dedicated format-quality pass, then validate + repair every
    cloze card locally so a broken/dead card can never reach save_cards().

    When lang == "uz", also runs a local Cyrillic safety net (see
    _enforce_uz_cyrillic below) for the same reason the cloze repair exists:
    the generation/quality prompts ask the model to write Cyrillic Uzbek,
    but nothing guarantees compliance short of checking locally.
    """
    if not cards:
        return [], FormatResult(cards_in=0, cards_out=0)

    if not groq_api_key:
        # No key — skip the AI pass but STILL run the local safety nets below,
        # since those cost nothing and catch the worst failure modes.
        result = FormatResult(cards_in=len(cards), cards_out=len(cards))
        cards  = _validate_and_repair(cards, result)
        if lang == "uz":
            cards = _enforce_uz_cyrillic(cards, result)
        return cards, result

    client   = Groq(api_key=groq_api_key)
    cards_in = len(cards)
    batches  = [cards[i:i + FORMAT_BATCH] for i in range(0, len(cards), FORMAT_BATCH)]

    print(f"\n  🎨  Format pass: {cards_in} cards in {len(batches)} batch(es) [Groq]...")

    all_formatted = []
    for i, batch in enumerate(batches, 1):
        for card in batch:
            if isinstance(card.get("back"),  list):
                card["back"]  = "\n".join(str(x) for x in card["back"])
            if isinstance(card.get("front"), list):
                card["front"] = " ".join(str(x) for x in card["front"])

        print(f"  🔄  Format batch {i}/{len(batches)}...", end=" ", flush=True)
        formatted = _format_batch(batch, deck, client)
        print(f"done ({len(formatted)} cards)")
        all_formatted += formatted

        if i < len(batches):
            time.sleep(2)

    result = FormatResult(cards_in=cards_in, cards_out=len(all_formatted))
    all_formatted = _validate_and_repair(all_formatted, result)
    if lang == "uz":
        all_formatted = _enforce_uz_cyrillic(all_formatted, result)

    print(f"\n  {result.summary()}")
    return all_formatted, result


def _validate_and_repair(cards: list[dict], result: FormatResult) -> list[dict]:
    """
    Local safety net — runs whether or not the AI pass ran. Guarantees that
    no "cloze" card without real {{c1::...}} syntax ever reaches save_cards().
    """
    for card in cards:
        front = card.get("front", "") or ""
        back  = card.get("back",  "") or ""

        # 1. Cloze validity check
        if card.get("type") == "cloze" and not _has_real_cloze(front):
            fixed = _convert_legacy_cloze(front)
            if fixed:
                card["front"] = fixed
                result.cloze_fixed += 1
            else:
                # Can't safely convert — a broken cloze card is worse than a
                # visible basic card, so demote it rather than lose it silently.
                card["type"] = "basic"
                result.cloze_demoted += 1

        # 2. Markdown → HTML safety net (in case the AI pass missed any)
        new_front, changed_f = _strip_markdown_bold(card.get("front", ""))
        new_back,  changed_b = _strip_markdown_bold(card.get("back",  ""))
        card["front"] = new_front
        card["back"]  = new_back
        if changed_f or changed_b:
            result.markdown_stripped += 1

    return cards


def _enforce_uz_cyrillic(cards: list[dict], result: FormatResult) -> list[dict]:
    """
    Local safety net for lang="uz" — guarantees Cyrillic script the same
    way _validate_and_repair() guarantees real cloze syntax: the prompt
    asks nicely, this makes it true regardless.

    Deliberately conservative: a field is only transliterated if it
    contains NO Cyrillic characters at all. If the model already wrote
    Cyrillic (even mixed with e.g. a kept Latin abbreviation), that field
    is left untouched rather than risk mangling something that's already
    correct. See uz_transliterate.py's docstring for the tradeoffs this
    implies — it's a best-effort net, not a guarantee that 100% of a card
    that ignored the language instruction entirely will read correctly.
    """
    for card in cards:
        for key in ("front", "back"):
            text = card.get(key, "") or ""
            if isinstance(text, list):
                text = " ".join(str(x) for x in text)
            if text and not has_cyrillic(text):
                card[key] = to_cyrillic(text)
                result.uz_transliterated += 1

    return cards
