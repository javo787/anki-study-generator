"""
integrity.py — a model may EDIT a card, it may never REPLACE one.

Before, the filter / quality / format passes asked a model to re-emit every card and used its
answer as the new deck: nothing checked the count, the ids or the content, unknown fields (the
source-link time) were lost, and a card the model forgot simply vanished.
"""
import copy

import pytest

import integrity
from integrity import merge_returned, number_changes, numbers_of, plain_text, same_content, similarity


def card(i, **kw):
    return {"front": f"Front {i}?", "back": f"Back {i}", "type": "basic", "tags": ["x"], "t": 100 + i, **kw}


# ── numbers and plain text ──────────────────────────────────────────────────────

def test_plain_text_drops_markup_and_cloze_syntax():
    assert plain_text("What?<br><br>{{c1::Dura <b>mater</b>}} (outer)") == "What? Dura mater (outer)"


def test_cloze_indices_are_not_numbers():
    assert numbers_of("{{c1::C3}}, {{c2::C4}} and C5 keep it alive 1.5 mg") == {"3": 1, "4": 1, "5": 1, "1.5": 1}


@pytest.mark.parametrize("before, after, removed, added", [
    ("C3,4,5 keeps the diaphragm alive", "C3-5 keeps the diaphragm alive", ["4"], []),
    ("dose 50 mg", "dose 150 mg", ["50"], ["150"]),
    ("dose 50 mg", "<b>dose</b> 50 mg", [], []),
    ("pH 7.4", "pH 7,4", ["7.4"], ["4", "7"]),
])
def test_number_changes(before, after, removed, added):
    assert number_changes([before], [after]) == (removed, added)


# ── same_content: what a FORMAT pass may do ─────────────────────────────────────

@pytest.mark.parametrize("before, after", [
    (("What produces CSF?", "**Choroid plexus**\n• ventricles"), ("What produces CSF?", "<b>Choroid plexus</b><br>• ventricles")),
    (("Name it ___ (dura)", "outer layer"), ("Name it {{c1::dura}}", "outer layer")),
    (("Q?", "Plural: die Ärzte"), ("Q?", "Plural: die Ärzte<br>")),
])
def test_formatting_only_changes_are_accepted(before, after):
    assert same_content(before, after)


@pytest.mark.parametrize("before, after", [
    (("Dose?", "50 mg daily"), ("Dose?", "150 mg daily")),                                   # a number changed
    (("Where is CSF made?", "choroid plexus of the ventricles"), ("Where is CSF made?", "arachnoid villi at the sagittal sinus")),
    (("Q?", "phrenic nerve C3 C4 C5"), ("Q?", "vagus nerve")),
])
def test_content_changes_are_rejected(before, after):
    assert not same_content(before, after)


def test_similarity_is_high_for_small_edits_and_low_for_rewrites():
    assert similarity("The phrenic nerve supplies the diaphragm", "The phrenic nerve supplies the diaphragm.") > 0.95
    assert similarity("The phrenic nerve supplies the diaphragm", "Tibial nerve damage causes foot drop") < 0.5


# ── merging what a model returned onto the ORIGINAL cards ───────────────────────

def returned(i, **kw):
    return {"id": i, "front": f"New front {i}?", "back": f"New back {i}", "type": "cloze", "tags": ["evil"], **kw}


def test_changes_are_merged_by_id_onto_the_original_cards():
    originals = [card(0), card(1), card(2)]
    snapshot = copy.deepcopy(originals)
    info = merge_returned(originals, [returned(2), returned(0)], ("front", "back"))     # reordered, one missing
    assert [c["front"] for c in info.cards] == ["New front 0?", "Front 1?", "New front 2?"]
    assert originals == snapshot                                                          # the originals are never mutated
    assert info.matched == 2 and info.missing == 1 and not info.rejected and not info.positional


def test_everything_the_model_was_not_asked_to_change_stays_as_it_was():
    info = merge_returned([card(0)], [returned(0)], ("front", "back"))
    merged = info.cards[0]
    assert merged["type"] == "basic" and merged["tags"] == ["x"] and merged["t"] == 100     # type, tags and the source time survive


def test_a_card_the_model_forgot_is_not_lost():
    info = merge_returned([card(0), card(1), card(2), card(3)], [returned(1)], ("front", "back"))
    assert len(info.cards) == 4 and info.cards[0] == card(0) and info.missing == 3


def test_unknown_duplicate_and_malformed_entries_are_ignored():
    items = [returned(9), returned(1), returned(1, front="second answer for the same id"), "junk", None, {"front": "no id"}, returned(True)]
    info = merge_returned([card(0), card(1)], items, ("front", "back"))
    assert info.cards[1]["front"] == "New front 1?" and info.cards[0] == card(0)
    assert info.matched == 1 and info.ignored >= 5


def test_string_ids_are_accepted():
    info = merge_returned([card(0), card(1)], [returned("1")], ("front", "back"))
    assert info.cards[1]["front"] == "New front 1?"


def test_lists_are_flattened_and_blank_fields_ignored():
    r = {"id": 0, "front": ["Q", "part two"], "back": "   "}
    info = merge_returned([card(0)], [r], ("front", "back"))
    assert info.cards[0]["front"] == "Q part two" and info.cards[0]["back"] == "Back 0"


def test_without_ids_the_same_number_of_cards_is_matched_by_position():
    items = [{k: v for k, v in returned(i).items() if k != "id"} for i in range(3)]
    info = merge_returned([card(0), card(1), card(2)], items, ("front", "back"))
    assert info.positional and [c["front"] for c in info.cards] == ["New front 0?", "New front 1?", "New front 2?"]


def test_without_ids_and_a_different_count_the_whole_answer_is_rejected():
    items = [{k: v for k, v in returned(i).items() if k != "id"} for i in range(2)]
    originals = [card(0), card(1), card(2)]
    info = merge_returned(originals, items, ("front", "back"))
    assert info.rejected and info.cards == originals


@pytest.mark.parametrize("junk", [None, "text", {"cards": []}, [], 5])
def test_garbage_answers_change_nothing(junk):
    originals = [card(0), card(1)]
    info = merge_returned(originals, junk, ("front", "back"))
    assert info.rejected and info.cards == originals and info.matched == 0
