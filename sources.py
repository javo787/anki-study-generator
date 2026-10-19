"""
sources.py — "where in the video was this said?"

Every card made from a YouTube video gets a link back to the moment in the lecture,
so a confusing card in Anki is one click away from the explanation.

How the time is found (no extra tokens, nothing the model can make up):

  1. locate_cards():  right after a chunk is generated, each card is matched against
     the captions of ITS chunk. The best 3-caption window wins; words that occur in
     many captions count for little, rare words for a lot. No confident match (for
     example cards written in another language than the lecture) -> the start of the
     chunk, which is coarse but never wrong.
  2. restore_times(): the filter / quality / format passes let a model rewrite the
     cards and it may drop fields it does not know. The time is therefore matched
     back to the original cards by text similarity before the file is written.
  3. source_html(): only when saving, the time becomes a link appended to the back.

The card carries a single number, "t" (whole seconds from the start of the video).
"""

import math
from collections import Counter

from textmatch import card_text, content_tokens, jaccard

LEAD_IN_SECONDS     = 5      # start a little before the caption so the sentence is not cut
WINDOW_ENTRIES      = 3      # a card summarises a few consecutive captions (~10-15 s)
MIN_SHARED_TOKENS   = 2      # one coincidental word never decides a position
MIN_SCORE           = 1.0    # summed rarity (1 / number of captions containing the word)
RESTORE_MIN_JACCARD = 0.3    # how alike a rewritten card must be to its original


# ── small helpers ───────────────────────────────────────────────────────────────

def format_time(seconds) -> str:
    """303 -> '05:03', 3725 -> '1:02:05'."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def video_url(video_id: str, seconds) -> str:
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def is_valid_time(t) -> bool:
    """A usable 't': a real, finite, non-negative number (not a bool, not '05:03')."""
    return (isinstance(t, (int, float)) and not isinstance(t, bool)
            and math.isfinite(t) and t >= 0)


def source_html(video_id: str, seconds) -> str:
    """The link as it is appended to a card (HTML — the import file sets #html:true)."""
    return f'<small>▶ <a href="{video_url(video_id, seconds)}">{format_time(seconds)}</a></small>'


# ── 1. locate a card inside its chunk ───────────────────────────────────────────

def _best_start(card_tokens: set, entries: list, entry_tokens: list, df: Counter):
    """Start time of the best-matching caption window, or None if there is no confident match."""
    n = len(entries)
    shared = [card_tokens & toks for toks in entry_tokens]

    best_score, best_first = 0.0, None
    for i in range(n):
        window = shared[i:i + WINDOW_ENTRIES]
        union = set().union(*window)
        if len(union) < MIN_SHARED_TOKENS:
            continue
        score = sum(1.0 / df[t] for t in union)
        if score > best_score + 1e-9:                       # strictly better: ties keep the earliest window
            best_score = score
            best_first = i + next(k for k, s in enumerate(window) if s)   # first caption that really matches
    if best_first is None or best_score < MIN_SCORE:
        return None
    return entries[best_first]["start"]


def locate_cards(cards: list, chunk) -> None:
    """Set card['t'] (int seconds) on every card generated from `chunk` (a YouTube chunk)."""
    entries      = list(getattr(chunk, "entries", None) or [])
    entry_tokens = [content_tokens(e["text"]) for e in entries]
    df           = Counter(t for toks in entry_tokens for t in toks)
    fallback     = int(max(0, chunk.start_time))

    for card in cards:
        start = _best_start(content_tokens(card_text(card)), entries, entry_tokens, df) if entries else None
        card["t"] = fallback if start is None else int(max(0, start - LEAD_IN_SECONDS))


# ── 2. get the time back after the cards were rewritten ─────────────────────────

def restore_times(final_cards: list, raw_cards: list) -> int:
    """
    Give every card in `final_cards` a valid 't'. A card that already has one keeps it;
    otherwise the most similar card in `raw_cards` (the ones locate_cards() stamped,
    before any model rewrote them) lends its time. Returns how many cards got a time
    this way; cards that resemble nothing are left without one.
    """
    originals = [(content_tokens(card_text(c)), c["t"]) for c in raw_cards if is_valid_time(c.get("t"))]
    assigned = 0

    for card in final_cards:
        if is_valid_time(card.get("t")):
            card["t"] = int(card["t"])
            continue
        card.pop("t", None)
        tokens = content_tokens(card_text(card))
        best_sim, best_t = 0.0, None
        for orig_tokens, t in originals:
            sim = jaccard(tokens, orig_tokens)
            if sim > best_sim:
                best_sim, best_t = sim, t
        if best_t is not None and best_sim >= RESTORE_MIN_JACCARD:
            card["t"] = int(best_t)
            assigned += 1
    return assigned
