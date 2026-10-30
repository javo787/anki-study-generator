"""The quality filter answers with ids to REMOVE; it can no longer rewrite or re-emit the deck."""
import json
import types

import pytest

import filter as flt


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(flt.time, "sleep", lambda s: None)


class Groq:
    """Records the request; answers with `reply` (str, dict or callable(payload) -> str|dict) or raises it."""
    def __init__(self, reply):
        self.reply, self.requests = reply, []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model, messages, temperature):
        user = messages[1]["content"]
        payload = json.loads(user[user.index("Cards:\n") + 7: user.rindex("\n\nReturn")])
        self.requests.append((messages[0]["content"], payload))
        out = self.reply(payload) if callable(self.reply) else self.reply
        if isinstance(out, Exception):
            raise out
        out = out if isinstance(out, str) else json.dumps(out)
        msg = types.SimpleNamespace(content=out)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def deck(n=6):
    return [{"front": f"Question {i}?", "back": f"Answer {i}", "type": "basic", "tags": ["x"], "t": 30 * i} for i in range(n)]


def run(cards, reply):
    client = Groq(reply)
    kept, removed = flt._filter_batch(cards, "D", client)
    return kept, removed, client


def test_cards_are_sent_with_ids_and_without_the_fields_the_model_has_no_use_for():
    cards = deck(3)
    _, _, client = run(cards, {"removed": []})
    system, payload = client.requests[0]
    assert [p["id"] for p in payload] == [0, 1, 2]
    assert set(payload[0]) == {"id", "front", "back", "type"}                    # no tags, no source time
    assert "ONLY the cards to remove" in system and '"removed"' in system and '"kept"' not in system


def test_only_the_listed_ids_are_removed_and_the_survivors_are_the_same_objects():
    cards = deck(6)
    kept, removed, _ = run(cards, {"removed": [{"id": 1, "reason": "trivial"}, {"id": 4, "reason": "too_vague"}]})
    assert [c["front"] for c in kept] == ["Question 0?", "Question 2?", "Question 3?", "Question 5?"]
    assert all(k is cards[i] for k, i in zip(kept, [0, 2, 3, 5]))                # identical objects: tags and `t` untouched
    assert removed == [{"front": "Question 1?", "reason": "trivial"}, {"front": "Question 4?", "reason": "too_vague"}]


def test_an_answer_in_the_old_style_cannot_rewrite_the_deck():
    cards = deck(4)
    old_style = {"kept": [{"front": "HALLUCINATED", "back": "x", "type": "basic", "tags": []}], "removed": [{"front": "Question 2?", "reason": "trivial"}]}
    kept, removed, _ = run(cards, old_style)
    assert kept == cards and removed == []


def test_plain_integers_and_unknown_things_are_handled():
    cards = deck(6)
    kept, removed, _ = run(cards, {"removed": [2, {"id": 3, "reason": "because"}, {"id": 99, "reason": "trivial"},
                                               {"id": 2, "reason": "duplicate"}, "junk", {"reason": "no id"}, {"id": True}]})
    assert [c["front"] for c in kept] == ["Question 0?", "Question 1?", "Question 4?", "Question 5?"]
    assert [r["reason"] for r in removed] == ["unknown", "unknown"]              # first mention of an id wins; unknown reason -> unknown


@pytest.mark.parametrize("reply", ["sorry, I cannot do that", "", "{}", {"removed": "all of them"}, {"cards": []}, RuntimeError("boom 500")])
def test_anything_unusable_keeps_every_card(reply):
    cards = deck(5)
    kept, removed, _ = run(cards, reply)
    assert kept == cards and removed == []


def test_an_implausible_mass_removal_is_ignored_and_reported():
    cards = deck(10)
    warnings = []
    client = Groq({"removed": [{"id": i, "reason": "trivial"} for i in range(8)]})
    kept, removed = flt._filter_batch(cards, "D", client, warnings)
    assert kept == cards and removed == [] and len(warnings) == 1 and "8 of 10" in warnings[0]


def test_removing_most_of_a_small_batch_is_allowed():
    cards = deck(5)
    kept, removed, _ = run(cards, {"removed": [{"id": i, "reason": "trivial"} for i in range(4)]})
    assert len(kept) == 1 and len(removed) == 4


def test_rate_limit_is_retried(monkeypatch):
    cards, calls = deck(4), {"n": 0}

    def reply(payload):
        calls["n"] += 1
        return RuntimeError("429 rate limit") if calls["n"] == 1 else {"removed": [{"id": 0, "reason": "broken"}]}

    kept, removed, _ = run(cards, reply)
    assert calls["n"] == 2 and len(kept) == 3 and removed[0]["reason"] == "broken"


def test_filter_cards_reports_the_warning_and_the_reasons(monkeypatch):
    cards = deck(12)
    monkeypatch.setattr(flt, "Groq", lambda api_key: Groq({"removed": [{"id": 0, "reason": "trivial"}, {"id": 1, "reason": "trivial"}, {"id": 2, "reason": "broken"}]}))
    result = flt.filter_cards(cards, "D", "key", semantic_dedup=False)
    assert result.cards_in == 12 and result.cards_out == 9 and result.reasons == {"trivial": 2, "broken": 1}
    assert [c["front"] for c in result._kept_cards][:2] == ["Question 3?", "Question 4?"] and result.warnings == []

    monkeypatch.setattr(flt, "Groq", lambda api_key: Groq({"removed": [{"id": i, "reason": "trivial"} for i in range(11)]}))
    result = flt.filter_cards(deck(12), "D", "key", semantic_dedup=False)
    assert result.cards_out == 12 and len(result.warnings) == 1
