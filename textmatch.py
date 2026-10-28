"""
textmatch.py — language-agnostic content tokens for comparing short texts.

Shared by filter.py (duplicate cards) and sources.py (where in the video was this
said?). No heavy imports on purpose.

A "token" is a 5-character stem of a word with 3+ letters (so "produced" and
"produces" match), after HTML tags and cloze markup are removed. Works for any
script that has word characters — Latin, Cyrillic, German umlauts.
"""

import re

_HTML_TAG = re.compile(r"<[^>]+>")
_CLOZE    = re.compile(r"\{\{c\d+::(.*?)(?:::.*?)?\}\}", re.DOTALL)
_WORD     = re.compile(r"\w+", re.UNICODE)

STEM_LEN = 5


def as_text(value) -> str:
    """Card fields may be a string or (raw LLM output) a list of lines."""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    return str(value or "")


def card_text(card: dict) -> str:
    return f"{as_text(card.get('front'))} {as_text(card.get('back'))}"


def content_tokens(text: str) -> set:
    """Set of word stems in `text` (markup removed, lower-cased, numbers ignored)."""
    text = _CLOZE.sub(r"\1", as_text(text))
    text = _HTML_TAG.sub(" ", text).lower()
    return {w[:STEM_LEN] for w in _WORD.findall(text) if len(w) >= 3 and not w.isdigit()}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
