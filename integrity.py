"""
integrity.py — a model may EDIT a card, it may never REPLACE one.

The filter, quality and format passes used to ask a model to re-emit every card and then used
its answer AS the deck. Nothing checked the count, the ids or the content: a card the model
forgot simply vanished, unknown fields (the source-link time) were lost, and "fix medical
inaccuracies silently" changed medical text with no trace.

Now every pass works like this:
  * cards carry an id in the request (their position in the batch) and the model is asked to echo it;
  * the answer is MERGED onto the original card objects (merge_returned): only the fields the pass
    is allowed to change are taken over, a card the model dropped keeps its original, unknown or
    duplicate ids are ignored, and an answer that cannot be matched is rejected as a whole;
  * what changed is measured here, by the code (a diff, and the numbers in the text), not taken
    from what the model claims;
  * a format pass must not change content at all (same_content); a quality pass may, and every
    change is recorded and rendered into a report for the learner to review.

No imports from the rest of the project except textmatch (pure functions).
"""

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

from textmatch import as_text, content_tokens, jaccard, plain_text

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

SAME_CONTENT_JACCARD = 0.85     # how alike the words of a card must stay after a FORMAT pass


# ── what changed ────────────────────────────────────────────────────────────────

def numbers_of(*texts) -> Counter:
    """The numbers in some card text (doses, values, nerve roots ...). In medicine a changed
    number is the cheapest reliable signal that a FACT changed. Cloze indices (c1) are not numbers."""
    counts = Counter()
    for text in texts:
        counts.update(_NUMBER.findall(plain_text(text)))
    return counts


def number_changes(before_texts, after_texts) -> tuple:
    """(removed, added) numbers between two groups of texts, each sorted. Compared as SETS: a
    mnemonic that repeats "C3,4,5" does not change a number, a "50" that became "150" does."""
    before, after = set(numbers_of(*before_texts)), set(numbers_of(*after_texts))
    return sorted(before - after), sorted(after - before)


def similarity(before: str, after: str) -> float:
    """0..1 how alike two texts read (1 = identical, ignoring markup and case)."""
    a, b = plain_text(before).lower(), plain_text(after).lower()
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def same_content(before: tuple, after: tuple, min_jaccard: float = SAME_CONTENT_JACCARD) -> bool:
    """
    True if (front, back) `after` still says the same as `before` — only formatting changed.
    The numbers must be identical and the words (stems) must overlap by `min_jaccard`.
    """
    if set(numbers_of(*before)) != set(numbers_of(*after)):
        return False
    a = content_tokens(" ".join(as_text(x) for x in before))
    b = content_tokens(" ".join(as_text(x) for x in after))
    if not a and not b:
        return True
    return jaccard(a, b) >= min_jaccard


# ── merging a model's answer onto the original cards ────────────────────────────

@dataclass
class MergeInfo:
    cards:      list                                   # merged cards, parallel to the originals
    returned:   list = field(default_factory=list)     # the model's entry for each original (dict) or None
    matched:    int  = 0                               # originals the model returned something for
    missing:    int  = 0                               # originals the model did not return (kept as they were)
    ignored:    int  = 0                               # entries that were unknown / duplicate / malformed
    positional: bool = False                           # matched by position because the model sent no ids
    rejected:   bool = False                           # the answer could not be matched at all: nothing taken


def _as_id(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _flatten(value, joiner):
    if isinstance(value, list):
        value = joiner.join(str(x) for x in value)
    return value if isinstance(value, str) else None


def merge_returned(originals: list, returned, fields: tuple = ("front", "back")) -> MergeInfo:
    """
    Merge a model's answer onto `originals` (never mutated) and return a MergeInfo.

    Each original is matched to the entry carrying its id (its position in the request). Only
    `fields` are taken from the model (lists are flattened, blank values ignored); everything
    else — type, tags, the source time ... — stays exactly as in the original.
    """
    n = len(originals)
    nothing = MergeInfo(cards=list(originals), returned=[None] * n, rejected=True)
    if not isinstance(returned, list) or not returned:
        return nothing

    entries = [r for r in returned if isinstance(r, dict)]
    by_id, ignored = {}, len(returned) - len(entries)
    for r in entries:
        i = _as_id(r.get("id"))
        if i is None or not (0 <= i < n) or i in by_id:
            ignored += 1
            continue
        by_id[i] = r

    positional = False
    if not by_id:
        # the model sent no usable ids: trust the order only if the count matches exactly
        no_ids = all(_as_id(r.get("id")) is None for r in entries)
        if no_ids and len(entries) == n:
            by_id, positional, ignored = dict(enumerate(entries)), True, 0
        else:
            return nothing

    merged, back = [], []
    for i, original in enumerate(originals):
        r = by_id.get(i)
        if r is None:
            merged.append(original)
            back.append(None)
            continue
        card = dict(original)
        for f in fields:
            value = _flatten(r.get(f), "\n" if f == "back" else " ")
            if value is not None and value.strip():
                card[f] = value
        merged.append(card)
        back.append(r)
    return MergeInfo(cards=merged, returned=back, matched=len(by_id), missing=n - len(by_id),
                     ignored=ignored, positional=positional)


# ── describing and reporting what a QUALITY pass changed ────────────────────────

REWRITE_REVIEW_SIMILARITY = 0.5       # below this a card counts as rewritten, not edited
REVIEW_TAG = "needs_review"


def _norm(text) -> str:
    return " ".join(as_text(text).split())


def describe_change(original: dict, merged: dict, claim: dict = None):
    """
    What did a quality pass do to one card? None if front and back read the same afterwards.

    Everything here is MEASURED (a diff, the numbers in the text); the model's `claim`
    (fixed_fact / mnemonic_added / note) only labels the change. A change is flagged for the
    learner's review when the model says it corrected a fact, when a NUMBER changed (even if
    the model did not say so), or when the card was rewritten rather than edited.
    """
    bf, bb = as_text(original.get("front")), as_text(original.get("back"))
    af, ab = as_text(merged.get("front")), as_text(merged.get("back"))
    if _norm(bf) == _norm(af) and _norm(bb) == _norm(ab):
        return None

    claim = claim or {}
    removed, added = number_changes((bf, bb), (af, ab))
    sim   = similarity(f"{bf} {bb}", f"{af} {ab}")
    fixed = bool(claim.get("fixed_fact"))
    note  = str(claim.get("note") or "").strip() if isinstance(claim.get("note"), (str, int, float)) else ""

    kinds = []
    if claim.get("mnemonic_added"):
        kinds.append("mnemonic")
    if fixed:
        kinds.append("fact fix")
    elif removed or added:
        kinds.append("number change")          # a number moved and the model did not say it fixed a fact
    if not kinds:
        kinds.append("phrasing")

    reasons = []
    if fixed:
        reasons.append("fact corrected" + (f": {note}" if note else ""))
    if removed or added:
        reasons.append(f"numbers changed (removed: {', '.join(removed) or '—'}; added: {', '.join(added) or '—'})")
    if sim < REWRITE_REVIEW_SIMILARITY:
        reasons.append(f"heavily rewritten (similarity {sim:.2f})")

    return {
        "before_front": bf, "before_back": bb, "after_front": af, "after_back": ab,
        "kinds": kinds, "note": note, "numbers_removed": removed, "numbers_added": added,
        "similarity": round(sim, 3), "review": bool(reasons), "review_reasons": reasons,
    }


def add_review_tag(card: dict) -> None:
    """Tag a card `needs_review` once (tags may be missing or not a list)."""
    tags = card.get("tags")
    tags = list(tags) if isinstance(tags, list) else ([str(tags)] if tags else [])
    if REVIEW_TAG not in tags:
        tags.append(REVIEW_TAG)
    card["tags"] = tags


def _one_line(text: str) -> str:
    return _norm(text).replace("\n", " ⏎ ")


def _title(change: dict, width: int = 70) -> str:
    """The card, identified by its question: positions change when the deck is sorted/formatted."""
    text = plain_text(change.get("before_front", ""))
    return text if len(text) <= width else text[:width - 1].rstrip() + "…"


def render_changes_report(deck: str, changes: list, total_cards: int) -> str:
    """A Markdown report of what the quality pass changed ('' when it changed nothing)."""
    if not changes:
        return ""
    review = [c for c in changes if c["review"]]
    other  = [c for c in changes if not c["review"]]
    out = [
        f"# Quality-check changes — {deck}", "",
        f"{len(changes)} of {total_cards} cards were changed by the AI quality check; "
        + (f"{len(review)} need your review (they carry the tag `{REVIEW_TAG}` in Anki)." if review else "none of them needs review."),
        "", f"## Needs review ({len(review)})", "",
    ]
    if not review:
        out += ["_Nothing needs review._", ""]
    for c in review:
        out += [f"### \"{_title(c)}\" — {'; '.join(c['review_reasons'])}", "",
                "**Before**", f"- Front: {_one_line(c['before_front'])}", f"- Back: {_one_line(c['before_back'])}", "",
                "**After**", f"- Front: {_one_line(c['after_front'])}", f"- Back: {_one_line(c['after_back'])}", ""]
    out += [f"## Other changes ({len(other)})", ""]
    for c in other:
        out.append(f"- **\"{_title(c)}\"** ({', '.join(c['kinds'])})")
        if _norm(c["before_front"]) != _norm(c["after_front"]):
            out.append(f"  - Front: {_one_line(c['before_front'])} → {_one_line(c['after_front'])}")
        if _norm(c["before_back"]) != _norm(c["after_back"]):
            out.append(f"  - Back: {_one_line(c['before_back'])} → {_one_line(c['after_back'])}")
    out.append("")
    return "\n".join(out)
