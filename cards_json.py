"""
cards_json.py — tolerant parsing of the card JSON that LLMs return.

Used by anki_gen.py (Gemini) and generator.py (NVIDIA / Groq / OpenRouter),
which previously each had their own copy of a greedy-regex parser. That parser
threw away the WHOLE chunk when

  * the model added prose containing braces after the JSON ("... {see above}"),
    because the greedy regex (first "{" to last "}") swallowed the prose and
    json.loads() failed, or
  * the reply was cut off by the output-token limit — even if 20 of the 25
    cards were complete.

This module has no heavy imports on purpose, so it is trivially testable.
"""

import json
import re

_FENCE_START = re.compile(r"^```[A-Za-z]*\s*\n?")
_FENCE_END   = re.compile(r"\n?```\s*$")
_CARDS_KEY   = re.compile(r'"cards"\s*:\s*\[')
_SKIP        = re.compile(r"[\s,]*")

MAX_START_CANDIDATES = 50   # how many '{' / '[' we try as the start of the JSON


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = _FENCE_START.sub("", raw)
    raw = _FENCE_END.sub("", raw)
    return raw.strip()


def _is_card(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    front = obj.get("front")
    return bool(front) and isinstance(front, (str, list))


def _cards_from_value(value):
    """Return a list of cards if `value` looks like our payload, else None."""
    if isinstance(value, dict) and isinstance(value.get("cards"), list):
        return [c for c in value["cards"] if _is_card(c)]
    if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
        cards = [c for c in value if _is_card(c)]
        return cards or None
    return None


def _salvage(raw: str) -> list[dict]:
    """Collect every COMPLETE card object from a truncated '{"cards": [ ... ' reply."""
    decoder = json.JSONDecoder()
    m = _CARDS_KEY.search(raw)
    if m:
        pos = m.end()
    elif raw.startswith("["):
        pos = 1
    else:
        return []

    cards = []
    while True:
        pos = _SKIP.match(raw, pos).end()
        if pos >= len(raw) or raw[pos] != "{":
            break
        try:
            obj, pos = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError:
            break                      # the card that was cut off — drop it, keep the rest
        if _is_card(obj):
            cards.append(obj)
    return cards


def parse_cards_ex(raw: str) -> tuple[list[dict], bool]:
    """
    Parse a model reply into a list of card dicts.

    Returns (cards, salvaged). `salvaged` is True when the reply was not valid
    JSON as a whole and only the complete cards could be recovered.
    Returns ([], False) when nothing usable is found.
    Raises TypeError if `raw` is not a string (e.g. a blocked/empty response),
    so callers keep their existing "provider failed -> try the next one" path.
    """
    if not isinstance(raw, str):
        raise TypeError("model returned no text")

    raw = _strip_fences(raw)
    decoder = json.JSONDecoder()

    tried = 0
    for m in re.finditer(r"[{\[]", raw):
        if tried >= MAX_START_CANDIDATES:
            break
        tried += 1
        try:
            value, _end = decoder.raw_decode(raw, m.start())   # stops after the first complete value,
        except json.JSONDecodeError:                           # so trailing prose/braces are ignored
            continue
        cards = _cards_from_value(value)
        if cards is not None:
            return cards, False

    salvaged = _salvage(raw)
    return salvaged, bool(salvaged)


def parse_cards(raw: str) -> list[dict]:
    """Convenience wrapper: just the cards. Prints a notice if it had to salvage."""
    cards, salvaged = parse_cards_ex(raw)
    if salvaged:
        print(f"  ⚠️  Reply was cut off / malformed — salvaged {len(cards)} complete card(s)")
    return cards
