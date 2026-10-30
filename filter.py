"""
filter.py — Semantic card filtering using Groq Llama
Removes weak, duplicate, and low-quality cards.
Requires: pip install groq
"""

import re
import sys
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from cards_json import parse_json_object
from profiles import GERMAN_DEDUP_SYSTEM
from textmatch import as_text, card_text, content_tokens

# ── IMPORTS ────────────────────────────────────────────────────────────────────

try:
    from groq import Groq
except ImportError:
    sys.exit("❌  Run: pip install groq")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

GROQ_MODEL      = "llama-3.3-70b-versatile"
FILTER_BATCH    = 30    # cards per filtering request
MAX_RETRY       = 3
RETRY_WAIT      = 10

# ── DATA CLASSES ───────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    cards_in:     int
    cards_out:    int
    removed:      list[dict] = field(default_factory=list)
    reasons:      dict       = field(default_factory=dict)
    warnings:     list       = field(default_factory=list)   # e.g. an implausible mass removal that was ignored

    @property
    def removed_count(self) -> int:
        return self.cards_in - self.cards_out

    @property
    def kept_pct(self) -> float:
        if self.cards_in == 0:
            return 0.0
        return self.cards_out / self.cards_in * 100

    def summary(self) -> str:
        return (
            f"Filter: {self.cards_in} in → {self.cards_out} kept "
            f"({self.kept_pct:.0f}%) | "
            f"removed: {self.removed_count} | "
            f"reasons: {self.reasons}"
        )


# ── PROMPTS ────────────────────────────────────────────────────────────────────

FILTER_SYSTEM = """\
You are a strict medical Anki card quality reviewer.

Your job: review a batch of Anki flashcards (each has an "id") and point out the LOW-QUALITY ones.

REMOVE a card if ANY of these apply:
1. TOO VAGUE — front is a general definition ("What is X?", "Define X")
2. DUPLICATE — same fact as another card in the batch, even if worded differently
3. INCOMPLETE — answer is missing, too short, or says "see lecture"
4. TRIVIAL — tests obvious common knowledge a medical student already knows
5. BROKEN — front or back is garbled, cut off, or nonsensical

KEEP a card if ALL of these apply:
✓ Tests ONE specific fact (number, structure, pathway, clinical correlation)
✓ Front is a specific question (What are the branches of X? Where does X drain?)
✓ Back has a clear concise answer
✓ A medical student would genuinely benefit from this card

OUTPUT: JSON only — no markdown, no explanation.
List ONLY the cards to remove, by their "id". Every card you do not list is kept exactly as
it is — never rewrite, shorten or re-emit a card. Use an empty list when every card is good.
{"removed": [{"id": 12, "reason": "too_vague"}]}

Reason codes: too_vague | duplicate | incomplete | trivial | broken"""


FILTER_USER = """\
Review these {count} Anki cards and list the ids of the low-quality ones.
Deck: {deck}

Cards:
{cards_json}

Return JSON only — the ids to remove."""


# ── CORE FILTER ────────────────────────────────────────────────────────────────

FILTER_REASONS    = {"too_vague", "duplicate", "incomplete", "trivial", "broken"}
FILTER_MASS_MIN   = 8      # from this batch size on ...
FILTER_MASS_RATIO = 0.6    # ... removing more than this share is treated as a malfunction, not a verdict


def _filter_batch(
    cards:    list[dict],
    deck:     str,
    client:   Groq,
    warnings: list = None,
) -> tuple[list[dict], list[dict]]:
    """
    Send one batch to Groq for filtering. The model answers with the ids of the cards to
    REMOVE; the survivors are the original card objects, never the model's copies, so a card
    can neither be rewritten nor silently lost, and fields the model never saw (tags, the
    source-link time) are untouched.

    Fails open: any unusable answer keeps every card. Returns: (kept_cards, removed_cards)
    """
    payload = [
        {"id": i, "front": as_text(c.get("front")), "back": as_text(c.get("back")), "type": c.get("type", "basic")}
        for i, c in enumerate(cards)
    ]
    prompt = FILTER_USER.format(
        count      = len(cards),
        deck       = deck,
        cards_json = json.dumps(payload, indent=2, ensure_ascii=False),
    )

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model    = GROQ_MODEL,
                messages = [
                    {"role": "system", "content": FILTER_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature = 0.1,
            )
            data     = parse_json_object(response.choices[0].message.content)
            verdicts = data.get("removed") if isinstance(data, dict) else None
            if not isinstance(verdicts, list):
                print("  ⚠️  Filter answer unusable — keeping every card")
                return cards, []

            remove = {}                                   # id -> reason (the first mention of an id wins)
            for v in verdicts:
                i      = v.get("id") if isinstance(v, dict) else v
                reason = v.get("reason") if isinstance(v, dict) else None
                if isinstance(i, bool) or not isinstance(i, int) or not (0 <= i < len(cards)) or i in remove:
                    continue
                remove[i] = reason if reason in FILTER_REASONS else "unknown"

            if len(cards) >= FILTER_MASS_MIN and len(remove) > FILTER_MASS_RATIO * len(cards):
                msg = (f"The quality filter wanted to remove {len(remove)} of {len(cards)} cards in one batch — "
                       f"ignored as implausible, the batch was kept.")
                print(f"  ⚠️  {msg}")
                if warnings is not None:
                    warnings.append(msg)
                return cards, []

            kept    = [c for i, c in enumerate(cards) if i not in remove]
            removed = [{"front": as_text(cards[i].get("front")), "reason": r} for i, r in remove.items()]
            return kept, removed

        except Exception as e:
            err = str(e)
            if ("503" in err or "429" in err) and attempt < MAX_RETRY:
                wait = attempt * RETRY_WAIT
                print(f"  ⏳  Groq rate limit, retry {attempt}/{MAX_RETRY} in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ⚠️  Groq filter error: {err[:80]}")
                return cards, []   # on error — keep all

    return cards, []


# ── GLOBAL SEMANTIC DEDUP ──────────────────────────────────────────────────────
#
# The previous version cut the deck into batches of FILTER_BATCH cards and asked
# the model to dedupe each batch on its own, so two cards that restate the same
# fact in different parts of a lecture (card #3 and card #67) were never in the
# same request and were never compared. It also let the model re-emit every
# card's text and trusted the model's own "duplicates_removed" count.
#
# Now:
#   1. Cheap, deterministic, GLOBAL pass finds groups of suspiciously similar
#      cards (token overlap over the whole deck — no API calls).
#   2. Only those groups are sent to the model, which answers with card IDs
#      ("keep 12, remove 40") — it never rewrites card text, and an invalid or
#      garbled answer can only ever leave cards in, never corrupt them.

DEDUP_SIMILARITY   = 0.4   # Jaccard overlap above which two cards are "candidates"
DEDUP_MIN_SHARED   = 2     # ...and they must share at least this many content tokens
DEDUP_COMMON_FRAC  = 0.25  # tokens present in more than this share of cards carry no signal
DEDUP_COMMON_MIN_N = 12    # (only applied to decks at least this big)
DEDUP_GROUP_MAX    = 12    # max cards in one group sent to the model
DEDUP_REQUEST_MAX  = 40    # max cards per request (several groups are packed together)
DEDUP_PAUSE        = 2     # seconds between requests (rate limits)

DEDUP_SYSTEM = """\
You are reviewing Anki flashcards for semantic duplicates.

You get several GROUPS of cards that look alike. Each card has an integer id.
Inside a group, two cards are duplicates if they test the SAME fact, even if
worded differently. Cards that only share a topic but test DIFFERENT facts are
NOT duplicates — keep both.

Examples of duplicates:
  - "What nerve innervates the diaphragm?" and "Which nerve supplies the diaphragm?"
  - "Where is CSF produced?" and "What structure produces CSF?"

For each group keep the BETTER card (more specific question, cleaner answer) and
list the ids to remove. Use ONLY the ids you were given. Never write card text.

OUTPUT: JSON only — no markdown, no explanation. Use an empty "remove" list when
a group has no duplicates.
{"results": [{"group": 0, "keep": 12, "remove": [40]}]}"""


def _card_text(value) -> str:
    return as_text(value)


def _card_tokens(card: dict) -> set:
    """Language-agnostic content tokens (see textmatch.py)."""
    return content_tokens(card_text(card))


def _find_duplicate_groups(cards: list[dict]) -> list[list[int]]:
    """
    Indices of cards that look alike, grouped, across the WHOLE deck.
    No API calls. Groups are in deck order; each has 2..DEDUP_GROUP_MAX cards.
    """
    n      = len(cards)
    tokens = [_card_tokens(c) for c in cards]

    if n >= DEDUP_COMMON_MIN_N:
        freq = Counter(w for t in tokens for w in t)
        common = {w for w, k in freq.items() if k > DEDUP_COMMON_FRAC * n}
        tokens = [t - common for t in tokens]

    index = defaultdict(list)
    for i, t in enumerate(tokens):
        for w in t:
            index[w].append(i)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, t in enumerate(tokens):
        shared = Counter()
        for w in t:
            for j in index[w]:
                if j > i:
                    shared[j] += 1
        for j, k in shared.items():
            union = len(t) + len(tokens[j]) - k
            if k >= DEDUP_MIN_SHARED and k / union >= DEDUP_SIMILARITY:
                parent[find(j)] = find(i)

    components = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    groups = []
    for members in components.values():
        if len(members) < 2:
            continue
        for s in range(0, len(members), DEDUP_GROUP_MAX):      # very large cluster: slice it
            part = members[s:s + DEDUP_GROUP_MAX]
            if len(part) >= 2:
                groups.append(part)
    groups.sort(key=lambda g: g[0])
    return groups


def _ids_to_remove(verdict: dict, groups: dict[int, list[int]]) -> set:
    """
    Turn the model's verdict into a set of card ids to drop — defensively.
    `groups` maps group number -> the ids that were in that group of the request.
    Anything malformed is ignored, so the worst case is "nothing removed".
    """
    results = verdict.get("results") if isinstance(verdict, dict) else None
    if not isinstance(results, list):
        return set()

    keeps, removes = set(), set()
    for r in results:
        if not isinstance(r, dict):
            continue
        g, keep, remove = r.get("group"), r.get("keep"), r.get("remove")
        members = groups.get(g) if isinstance(g, int) else None
        if members is None or keep not in members or not isinstance(remove, list):
            continue
        keeps.add(keep)
        removes.update(i for i in remove if i in members and i != keep)

    return removes - keeps       # a card the model chose to keep is never removed


def _dedup_global(cards: list[dict], deck: str, client: Groq, system: str = None) -> list[dict]:
    """
    Final semantic deduplication over ALL cards (see the section comment above).
    Fails open: any error leaves the affected cards in the deck.
    """
    if len(cards) < 2:
        return cards

    system = system or DEDUP_SYSTEM
    groups = _find_duplicate_groups(cards)
    if not groups:
        print("  ✅  No duplicate candidates found — nothing to send to the model")
        return cards

    n_cand = sum(len(g) for g in groups)
    print(f"  🔍  {len(groups)} group(s) of similar cards ({n_cand} cards) out of {len(cards)}")

    # pack several groups into one request
    requests, current, size = [], [], 0
    for g in groups:
        if current and size + len(g) > DEDUP_REQUEST_MAX:
            requests.append(current)
            current, size = [], 0
        current.append(g)
        size += len(g)
    if current:
        requests.append(current)

    removed_ids = set()
    for r_no, req_groups in enumerate(requests, 1):
        numbered = {gi: g for gi, g in enumerate(req_groups)}
        payload  = [
            {"group": gi, "cards": [
                {"id": i, "front": _card_text(cards[i].get("front")), "back": _card_text(cards[i].get("back"))}
                for i in g
            ]}
            for gi, g in numbered.items()
        ]
        prompt = (
            f"Deck: {deck}\n\n"
            f"Groups:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
            f"Return JSON only."
        )
        print(f"  🔄  Dedup request {r_no}/{len(requests)}...", end=" ", flush=True)

        for attempt in range(1, MAX_RETRY + 1):
            try:
                response = client.chat.completions.create(
                    model       = GROQ_MODEL,
                    messages    = [{"role": "system", "content": system},
                                   {"role": "user",   "content": prompt}],
                    temperature = 0.1,
                )
                verdict = parse_json_object(response.choices[0].message.content)
                found   = _ids_to_remove(verdict, numbered)
                removed_ids |= found
                print(f"{len(found)} duplicate(s) removed")
                break
            except Exception as e:
                err = str(e)
                if ("503" in err or "429" in err) and attempt < MAX_RETRY:
                    wait = attempt * RETRY_WAIT
                    print(f"\n  ⏳  Groq rate limit, retry {attempt}/{MAX_RETRY} in {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"skipped ({e.__class__.__name__}) — keeping these cards")
                    break

        if r_no < len(requests):
            time.sleep(DEDUP_PAUSE)

    kept = [c for i, c in enumerate(cards) if i not in removed_ids]
    if removed_ids:
        print(f"  ✅  Dedup complete: {len(removed_ids)} duplicate(s) removed")
    return kept


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def filter_cards(
    cards:       list[dict],
    deck:        str,
    groq_api_key: str,
    semantic_dedup: bool = True,
    profile:     str = "medical",
) -> FilterResult:
    """
    Main entry point. Filter and deduplicate a list of cards.

    Args:
        cards:           raw cards from generator.py
        deck:            deck name for context
        groq_api_key:    Groq API key
        semantic_dedup:  run cross-batch semantic deduplication
        profile:         "medical" | "german". The quality filter below is medical
                         ("trivial for a medical student", "what is X? is vague") and
                         would remove vocabulary cards, so a german deck skips it and
                         only gets de-duplicated, with German rules.

    Returns:
        FilterResult with kept cards and statistics
    """
    if not cards:
        return FilterResult(cards_in=0, cards_out=0)

    client     = Groq(api_key=groq_api_key)
    cards_in   = len(cards)
    all_kept   = []
    all_removed = []
    reason_counts = {}
    warnings    = []

    # ── Step 1: quality filter in batches ─────────────────────────────────────
    batches = [
        cards[i:i + FILTER_BATCH]
        for i in range(0, len(cards), FILTER_BATCH)
    ]

    if profile == "german":
        print(f"\n  🔎  Quality filter skipped for the german profile (its rules are medical) — {cards_in} cards kept")
        batches = []
        all_kept.extend(cards)
    else:
        print(f"\n  🔎  Filtering {cards_in} cards in {len(batches)} batch(es)...")

    for i, batch in enumerate(batches, 1):
        print(f"  🔄  Filter batch {i}/{len(batches)}...", end=" ", flush=True)
        kept, removed = _filter_batch(batch, deck, client, warnings)
        print(f"{len(kept)} kept, {len(removed)} removed")

        all_kept.extend(kept)
        all_removed.extend(removed)

        # Count removal reasons
        for r in removed:
            reason = r.get("reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Small pause between batches to respect rate limits
        if i < len(batches):
            time.sleep(2)

    # ── Step 2: semantic deduplication across all kept cards ──────────────────
    if semantic_dedup and len(all_kept) > 1:
        print(f"\n  🔍  Semantic deduplication on {len(all_kept)} cards...")
        all_kept = _dedup_global(all_kept, deck, client, GERMAN_DEDUP_SYSTEM if profile == "german" else None)

    result = FilterResult(
        cards_in  = cards_in,
        cards_out = len(all_kept),
        removed   = all_removed,
        reasons   = reason_counts,
        warnings  = warnings,
    )

    print(f"\n  {result.summary()}")

    # Attach kept cards to result for downstream use
    result._kept_cards = all_kept
    return result


def get_kept_cards(result: FilterResult) -> list[dict]:
    """Extract kept cards from FilterResult."""
    return getattr(result, "_kept_cards", [])


# ── DEBUG ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        key = input("Groq API key: ").strip()

    # Test with sample cards
    test_cards = [
        {
            "front": "What are the three foramina of the fourth ventricle?",
            "back": "• Two foramina of Luschka (lateral)\n• One foramen of Magendie (median)",
            "type": "basic",
            "tags": ["neuroanatomy", "CSF"]
        },
        {
            "front": "What is CSF?",
            "back": "Cerebrospinal fluid",
            "type": "basic",
            "tags": ["neuroanatomy"]
        },
        {
            "front": "Name the openings of the fourth ventricle",
            "back": "• Foramina of Luschka x2\n• Foramen of Magendie x1",
            "type": "basic",
            "tags": ["neuroanatomy"]
        },
        {
            "front": "Where is CSF produced?",
            "back": "• Choroid plexuses\n• Lateral ventricles\n• Third and fourth ventricles",
            "type": "basic",
            "tags": ["neuroanatomy", "CSF"]
        },
    ]

    print(f"Testing filter with {len(test_cards)} cards...\n")
    result = filter_cards(test_cards, "Neuroanatomy - CSF", key)
    kept   = get_kept_cards(result)

    print(f"\nKept cards ({len(kept)}):")
    for c in kept:
        print(f"  Q: {c['front']}")
        print(f"  A: {c['back'][:60]}")
        print()