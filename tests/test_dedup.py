"""filter._dedup_global — GLOBAL, id-based semantic de-duplication."""
import json
import random

import pytest

import filter as flt


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(flt.time, "sleep", lambda *_: None)


def _noise_deck(n, seed=0):
    """n cards on unrelated made-up subjects (so they don't resemble each other)."""
    rng = random.Random(seed)
    syl = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
    word = lambda: "".join(rng.choice(syl) for _ in range(3))
    return [{"front": f"What is {word()} {word()}?", "back": f"{word()} {word()} {word()}",
             "type": "basic", "tags": []} for _ in range(n)]


def _deck_with_far_apart_duplicates():
    cards = _noise_deck(70)
    cards[2]  = {"front": "Where is CSF produced?", "back": "• Choroid plexus", "type": "basic", "tags": []}
    cards[66] = {"front": "Which structure produces CSF?", "back": "• Choroid plexus", "type": "basic", "tags": []}
    return cards


class FakeGroq:
    """Records each request; answers with verdict_fn(parsed_groups) -> dict | str."""
    def __init__(self, verdict_fn):
        self.requests, self.verdict_fn = [], verdict_fn
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, model, messages, temperature):
        user = messages[1]["content"]
        groups = json.loads(user[user.index("Groups:\n") + 8: user.rindex("\n\nReturn")])
        self.requests.append(groups)
        out = self.verdict_fn(groups)
        out = out if isinstance(out, str) else json.dumps(out)
        msg = type("M", (), {"content": out})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


def keep_first_remove_rest(groups):
    return {"results": [{"group": g["group"], "keep": g["cards"][0]["id"],
                         "remove": [c["id"] for c in g["cards"][1:]]} for g in groups]}


def test_candidates_are_found_across_the_whole_deck():
    cards = _deck_with_far_apart_duplicates()
    groups = flt._find_duplicate_groups(cards)
    assert groups == [[2, 66]]                      # 64 positions apart, i.e. different old batches


def test_far_apart_duplicate_is_actually_removed_and_order_is_kept():
    cards = _deck_with_far_apart_duplicates()
    client = FakeGroq(keep_first_remove_rest)
    kept = flt._dedup_global(cards, "Neuro", client)

    assert len(client.requests) == 1                # one cheap request, not one per 30-card batch
    assert len(kept) == 69
    fronts = [c["front"] for c in kept]
    assert "Where is CSF produced?" in fronts and "Which structure produces CSF?" not in fronts
    assert kept == [c for c in cards if c["front"] != "Which structure produces CSF?"]


def test_no_candidates_means_no_api_call():
    client = FakeGroq(keep_first_remove_rest)
    cards = _noise_deck(40, seed=1)
    assert flt._dedup_global(cards, "D", client) == cards
    assert client.requests == []


def test_cloze_markup_html_and_cyrillic_are_compared_by_content():
    a = {"front": "Қайси нерв диафрагмани таъминлайди?", "back": "• Френик нерв", "type": "basic", "tags": []}
    b = {"front": "Диафрагмани қайси нерв таъминлайди?<br>", "back": "<b>Френик</b> нерв", "type": "basic", "tags": []}
    c = {"front": "What makes CSF?<br><br>{{c1::Choroid plexus::structure}}", "back": "", "type": "cloze", "tags": []}
    d = {"front": "Which structure makes CSF?", "back": "Choroid plexus", "type": "basic", "tags": []}
    assert flt._find_duplicate_groups([a, b, c, d]) == [[0, 1], [2, 3]]


def test_different_facts_on_the_same_topic_are_not_grouped():
    cards = [
        {"front": "Which nerve supplies the diaphragm?", "back": "Phrenic nerve", "type": "basic", "tags": []},
        {"front": "What is the root value of the phrenic nerve?", "back": "C3, C4, C5", "type": "basic", "tags": []},
    ]
    # shares only 'phren'/'nerve' stems: below the similarity bar
    assert flt._find_duplicate_groups(cards) == []


# ── the model is never trusted blindly ─────────────────────────────────────────

def test_model_error_or_garbage_keeps_everything():
    cards = _deck_with_far_apart_duplicates()

    class Boom(FakeGroq):
        def create(self, **kw):
            raise RuntimeError("500 server exploded")

    assert flt._dedup_global(cards, "D", Boom(lambda g: {})) == cards
    assert flt._dedup_global(cards, "D", FakeGroq(lambda g: "sorry, I can't do that")) == cards


@pytest.mark.parametrize("verdict", [
    {"results": [{"group": 0, "keep": 999, "remove": [2, 66]}]},              # invented id
    {"results": [{"group": 0, "keep": 2, "remove": [5]}]},                    # id from another group
    {"results": [{"group": 7, "keep": 2, "remove": [66]}]},                   # unknown group
    {"results": [{"group": 0, "keep": "2", "remove": ["66"]}]},               # wrong types
    {"results": "all of them"},
    {"cards": []},
])
def test_invalid_verdicts_remove_nothing(verdict):
    cards = _deck_with_far_apart_duplicates()
    assert flt._dedup_global(cards, "D", FakeGroq(lambda g: verdict)) == cards


def test_contradictory_verdict_can_never_remove_both_cards():
    cards = _deck_with_far_apart_duplicates()
    verdict = {"results": [{"group": 0, "keep": 2, "remove": [66]},
                           {"group": 0, "keep": 66, "remove": [2]}]}
    kept = flt._dedup_global(cards, "D", FakeGroq(lambda g: verdict))
    assert len(kept) == len(cards)                  # both are 'kept' by someone -> nothing removed


def test_removing_the_kept_card_in_its_own_group_is_ignored():
    cards = _deck_with_far_apart_duplicates()
    verdict = {"results": [{"group": 0, "keep": 2, "remove": [2, 66]}]}
    kept = flt._dedup_global(cards, "D", FakeGroq(lambda g: verdict))
    assert [c["front"] for c in kept].count("Where is CSF produced?") == 1
    assert len(kept) == 69


def test_rate_limit_is_retried_then_succeeds():
    cards = _deck_with_far_apart_duplicates()
    calls = {"n": 0}

    class Flaky(FakeGroq):
        def create(self, model, messages, temperature):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("429 Too Many Requests")
            return super().create(model, messages, temperature)

    kept = flt._dedup_global(cards, "D", Flaky(keep_first_remove_rest))
    assert calls["n"] == 2 and len(kept) == 69


def test_groups_are_packed_into_few_requests():
    cards = _noise_deck(60, seed=2)
    # 30 made-up words whose 5-letter stems are all different
    words = ["kabuzo", "limepa", "nosiru", "tevagi", "pudoxe", "gizafo", "hoyuna", "jecivo", "qumawe", "xarenu",
             "bofetu", "cidalo", "damiko", "fenuba", "gorepi", "hutake", "iloxam", "jivosu", "kedapy", "lozuri",
             "mabeto", "nifoqa", "ovitle", "pukone", "rasiwa", "sedigu", "tomaxe", "uvelor", "wagifi", "yotuna"]
    assert len({w[:5] for w in words}) == 30
    for k in range(10):                              # 10 duplicate pairs scattered over the deck
        a, b, c = words[3 * k: 3 * k + 3]
        cards[k]      = {"front": f"Where is {a} made?",          "back": f"{b} {c}", "type": "basic", "tags": []}
        cards[30 + k] = {"front": f"Which {a} is made where?",    "back": f"{b} {c}", "type": "basic", "tags": []}
    client = FakeGroq(keep_first_remove_rest)
    kept = flt._dedup_global(cards, "D", client)
    assert len(client.requests) == 1                 # 20 candidate cards <= DEDUP_REQUEST_MAX
    assert len(kept) == 50


def test_filter_cards_end_to_end_uses_global_dedup(monkeypatch):
    """Whole public entry point with a fake Groq: quality filter keeps everything,
    dedup removes the far-apart duplicate."""
    class FakeGroqClient(FakeGroq):
        def __init__(self, api_key=None):
            super().__init__(keep_first_remove_rest)

        def create(self, model, messages, temperature):
            if messages[0]["content"] == flt.FILTER_SYSTEM:      # quality-filter request
                user = messages[1]["content"]
                cards = json.loads(user[user.index("Cards:\n") + 7: user.rindex("\n\nReturn")])
                msg = type("M", (), {"content": json.dumps({"kept": cards, "removed": []})})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
            return super().create(model, messages, temperature)

    monkeypatch.setattr(flt, "Groq", FakeGroqClient)
    cards = _deck_with_far_apart_duplicates()
    result = flt.filter_cards(cards, "Neuro", "fake-key")
    kept = flt.get_kept_cards(result)
    assert result.cards_in == 70 and result.cards_out == 69 == len(kept)
    assert "Which structure produces CSF?" not in [c["front"] for c in kept]
