"""
filter.py — Semantic card filtering using Groq Llama
Removes weak, duplicate, and low-quality cards.
Requires: pip install groq
"""

import re
import sys
import json
import time
from dataclasses import dataclass, field

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

Your job: review a batch of Anki flashcards and return only the HIGH-QUALITY ones.

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
{
  "kept": [
    {"front": "...", "back": "...", "type": "basic", "tags": [...], "score": 8}
  ],
  "removed": [
    {"front": "...", "reason": "too_vague"}
  ]
}

Reason codes: too_vague | duplicate | incomplete | trivial | broken"""


FILTER_USER = """\
Review these {count} Anki cards and return only the high-quality ones.
Deck: {deck}

Cards:
{cards_json}

Return JSON only."""


# ── CORE FILTER ────────────────────────────────────────────────────────────────

def _filter_batch(
    cards:  list[dict],
    deck:   str,
    client: Groq,
) -> tuple[list[dict], list[dict]]:
    """
    Send one batch to Groq for filtering.
    Returns: (kept_cards, removed_cards)
    """
    cards_json = json.dumps(cards, indent=2, ensure_ascii=False)
    prompt     = FILTER_USER.format(
        count     = len(cards),
        deck      = deck,
        cards_json = cards_json,
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
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data    = json.loads(raw)
            kept    = data.get("kept",    [])
            removed = data.get("removed", [])
            return kept, removed

        except json.JSONDecodeError as e:
            print(f"  ⚠️  Filter JSON parse error: {e}")
            return cards, []   # on parse error — keep all, don't lose cards

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


# ── CROSS-BATCH DEDUP ──────────────────────────────────────────────────────────

DEDUP_SYSTEM = """\
You are reviewing Anki flashcards for semantic duplicates.

Two cards are duplicates if they test the SAME medical fact, even if worded differently.
Examples of duplicates:
  - "What nerve innervates the diaphragm?" and "Which nerve supplies the diaphragm?"
  - "Where is CSF produced?" and "What structure produces CSF?"

Keep the BETTER card (more specific question, cleaner answer).

OUTPUT: JSON only — no markdown.
{
  "kept": [{"front": "...", "back": "...", "type": "...", "tags": [...]}],
  "duplicates_removed": 5
}"""


def _dedup_across_batches(
    cards:  list[dict],
    deck:   str,
    client: Groq,
) -> list[dict]:
    """
    Final semantic deduplication across all cards.
    Sends in batches of FILTER_BATCH to avoid token limits.
    """
    if len(cards) <= FILTER_BATCH:
        batches = [cards]
    else:
        batches = [
            cards[i:i + FILTER_BATCH]
            for i in range(0, len(cards), FILTER_BATCH)
        ]

    all_kept = []
    total_removed = 0

    for i, batch in enumerate(batches, 1):
        print(f"  🔍  Dedup batch {i}/{len(batches)}...", end=" ", flush=True)
        prompt = (
            f"Remove semantic duplicates from these {len(batch)} cards.\n"
            f"Deck: {deck}\n\n"
            f"Cards:\n{json.dumps(batch, indent=2, ensure_ascii=False)}\n\n"
            f"Return JSON only."
        )
        try:
            response = client.chat.completions.create(
                model    = GROQ_MODEL,
                messages = [
                    {"role": "system", "content": DEDUP_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature = 0.1,
            )
            raw  = response.choices[0].message.content.strip()
            raw  = re.sub(r"^```(?:json)?\s*", "", raw)
            raw  = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)

            kept    = data.get("kept", batch)
            removed = data.get("duplicates_removed", 0)
            total_removed += removed
            all_kept.extend(kept)
            print(f"{len(kept)} kept, {removed} removed")

        except Exception as e:
            print(f"skipped ({e.__class__.__name__})")
            all_kept.extend(batch)   # on error keep all

    if total_removed:
        print(f"  ✅  Dedup complete: {total_removed} duplicates removed")

    return all_kept


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def filter_cards(
    cards:       list[dict],
    deck:        str,
    groq_api_key: str,
    semantic_dedup: bool = True,
) -> FilterResult:
    """
    Main entry point. Filter and deduplicate a list of cards.

    Args:
        cards:           raw cards from generator.py
        deck:            deck name for context
        groq_api_key:    Groq API key
        semantic_dedup:  run cross-batch semantic deduplication

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

    # ── Step 1: quality filter in batches ─────────────────────────────────────
    batches = [
        cards[i:i + FILTER_BATCH]
        for i in range(0, len(cards), FILTER_BATCH)
    ]

    print(f"\n  🔎  Filtering {cards_in} cards in {len(batches)} batch(es)...")

    for i, batch in enumerate(batches, 1):
        print(f"  🔄  Filter batch {i}/{len(batches)}...", end=" ", flush=True)
        kept, removed = _filter_batch(batch, deck, client)
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
        all_kept = _dedup_across_batches(all_kept, deck, client)

    result = FilterResult(
        cards_in  = cards_in,
        cards_out = len(all_kept),
        removed   = all_removed,
        reasons   = reason_counts,
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