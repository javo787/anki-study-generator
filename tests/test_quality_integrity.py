"""
The quality pass may change a card — but never silently, never by replacing it, and what changed
is measured by the code instead of being taken from the model's own claims.
"""
import json
import types

import pytest

import integrity
import profiles
import quality


class FakeQuality:
    """Gemini stand-in: script(payload) -> list of returned cards | str | Exception."""
    def __init__(self, script):
        self.script, self.requests, self.systems = script, [], []
        self.models = self

    def generate_content(self, model, contents, config=None):
        payload = json.loads(contents[contents.index("Cards:\n") + 7: contents.rindex("\n\nReturn")])
        self.requests.append(payload)
        self.systems.append(config["system_instruction"])
        out = self.script(payload)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps({"cards": out})
        return types.SimpleNamespace(text=text)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(quality.time, "sleep", lambda s: None)


def use(monkeypatch, script):
    fake = FakeQuality(script)
    monkeypatch.setattr(quality.genai, "Client", lambda api_key: fake)
    return fake


def cards():
    mk = lambda f, b, t: {"front": f, "back": b, "type": "basic", "tags": ["neuro"], "t": t}
    return [mk("Which nerve supplies the diaphragm?", "Phrenic nerve", 10),
            mk("What are the nerve roots of the phrenic nerve?", "• C3\n• C4\n• C5", 20),
            mk("Which cranial nerves carry parasympathetics?", "III, VII, IX, X", 30),
            mk("What is the normal adult CSF volume?", "150 mL", 40),
            mk("Which structure reabsorbs CSF?", "Choroid plexus", 50)]


def returned(payload, overrides=None):
    """What a well-behaved model returns: every card with its id, unchanged, plus per-id edits."""
    out = [dict(p, improved=False, mnemonic_added=False) for p in payload]
    for i, edit in (overrides or {}).items():
        out[i].update(edit)
    return out


EDITS = {
    0: {"front": "Which nerve supplies the diaphragm muscle?", "improved": True},
    1: {"back": "• C3\n• C4\n• C5<br>💡 C3,4,5 keeps the diaphragm alive", "improved": True, "mnemonic_added": True},
    2: {"improved": True},                                                       # claims an improvement, changed nothing
    3: {"back": "100 mL"},                                                       # a silent change of a number
    4: {"back": "Arachnoid villi", "fixed_fact": True, "note": "villi reabsorb CSF, the plexus makes it"},
}


def run(monkeypatch, script=lambda p: returned(p, EDITS), **kw):
    fake = use(monkeypatch, script)
    out, res = quality.improve_cards(cards(), "Neuro", api_key="k", use_pro=False, **kw)
    return out, res, fake


# ── what is measured, what is flagged ───────────────────────────────────────────

def test_statistics_come_from_the_diff_not_from_the_models_claims(monkeypatch):
    out, res, _ = run(monkeypatch)
    assert res.mnemonics_added == 1 and res.phrasing_improved == 1             # card 2 only CLAIMED an improvement
    assert [c["index"] for c in res.changes] == [0, 1, 3, 4]


def test_changed_numbers_and_corrected_facts_are_flagged_for_review(monkeypatch):
    out, res, _ = run(monkeypatch)
    assert res.needs_review == 2
    tagged = [i for i, c in enumerate(out) if "needs_review" in c["tags"]]
    assert tagged == [3, 4]                                                       # the silent number change is caught too
    by_index = {c["index"]: c for c in res.changes}
    assert by_index[3]["numbers_removed"] == ["150"] and by_index[3]["numbers_added"] == ["100"]
    assert by_index[3]["kinds"] == ["number change"] and "numbers changed" in by_index[3]["review_reasons"][0]
    assert "fact" in by_index[4]["kinds"][0] and "villi reabsorb CSF" in by_index[4]["note"]
    assert not by_index[0]["review"] and not by_index[1]["review"]                # a new sentence and a mnemonic repeating C3,4,5


def test_the_before_and_after_text_is_recorded(monkeypatch):
    _, res, _ = run(monkeypatch)
    c = {x["index"]: x for x in res.changes}[3]
    assert (c["before_front"], c["before_back"]) == ("What is the normal adult CSF volume?", "150 mL")
    assert (c["after_front"], c["after_back"]) == ("What is the normal adult CSF volume?", "100 mL")


def test_a_heavy_rewrite_is_flagged_even_without_a_number(monkeypatch):
    out, res, _ = run(monkeypatch, lambda p: returned(p, {0: {"front": "Tibial nerve lesion gives which deformity?", "back": "Foot drop"}}))
    assert res.needs_review == 1 and "heavily rewritten" in res.changes[0]["review_reasons"][0]


# ── the deck is never replaced ──────────────────────────────────────────────────

def test_a_card_the_model_forgot_keeps_its_original(monkeypatch):
    out, res, _ = run(monkeypatch, lambda p: returned(p, EDITS)[:2])
    assert len(out) == 5 and res.cards_out == 5
    assert out[3]["back"] == "150 mL" and out[4]["back"] == "Choroid plexus"      # forgotten -> untouched, not "changed"


def test_type_tags_and_the_source_time_cannot_be_changed_by_the_model(monkeypatch):
    out, _, _ = run(monkeypatch, lambda p: returned(p, {0: {"type": "cloze", "tags": ["evil"], "t": 9999, "extra": "x", "front": "Which nerve supplies the diaphragm muscle?"}}))
    assert out[0]["type"] == "basic" and out[0]["tags"] == ["neuro"] and out[0]["t"] == 10 and "extra" not in out[0]
    assert all(c["t"] == 10 * (i + 1) for i, c in enumerate(out))


def test_no_ids_but_the_right_number_of_cards_is_matched_by_position(monkeypatch):
    out, res, _ = run(monkeypatch, lambda p: [{k: v for k, v in c.items() if k != "id"} for c in returned(p, EDITS)])
    assert out[0]["front"].endswith("muscle?") and res.rejected_batches == 0


def test_no_ids_and_the_wrong_number_of_cards_changes_nothing(monkeypatch):
    out, res, _ = run(monkeypatch, lambda p: [{k: v for k, v in c.items() if k != "id"} for c in returned(p, EDITS)][:3])
    assert [c["front"] for c in out] == [c["front"] for c in cards()] and res.rejected_batches == 1 and res.changes == []


@pytest.mark.parametrize("junk", ["I cannot help with that", "{}", '{"cards": "none"}', RuntimeError("500 boom")])
def test_garbage_or_failures_keep_every_card(monkeypatch, junk):
    out, res, _ = run(monkeypatch, lambda p: junk)
    assert out == cards() and res.changes == [] and res.needs_review == 0


# ── the request ─────────────────────────────────────────────────────────────────

def test_cards_are_sent_with_ids_and_without_the_source_time(monkeypatch):
    _, _, fake = run(monkeypatch)
    payload = fake.requests[0]
    assert [p["id"] for p in payload] == [0, 1, 2, 3, 4] and all("t" not in p for p in payload)
    assert set(payload[0]) == {"id", "front", "back", "type", "tags"}


@pytest.mark.parametrize("template", [quality.QUALITY_SYSTEM, profiles.GERMAN_QUALITY_SYSTEM])
def test_both_quality_prompts_demand_ids_and_an_admission_of_every_corrected_fact(template):
    text = template.format(language_instruction="X")
    assert '"id"' in text and '"fixed_fact"' in text and '"note"' in text
    assert "inaccuracies silently" not in text                                       # the old "fix ... silently" is gone


def test_changes_carry_global_indexes_across_batches(monkeypatch):
    monkeypatch.setattr(quality, "QUALITY_BATCH", 2)
    fronts = [c["front"] for c in cards()]

    def by_card(payload):                          # the ids in a request are batch-local; the edits belong to the cards
        out = returned(payload)
        for entry in out:
            edit = EDITS.get(fronts.index(entry["front"]))
            if edit:
                entry.update(edit)
        return out

    out, res, fake = run(monkeypatch, by_card)
    assert len(fake.requests) == 3 and [len(r) for r in fake.requests] == [2, 2, 1]
    assert [c["index"] for c in res.changes] == [0, 1, 3, 4]                       # not 0,1,1,0 (batch-local)
    assert res.needs_review == 2


def test_the_review_tag_is_added_once(monkeypatch):
    fake = use(monkeypatch, lambda p: returned(p, EDITS))
    cs = cards()
    for _ in range(2):
        cs, _ = quality.improve_cards(cs, "Neuro", api_key="k", use_pro=False)
    assert cs[3]["tags"].count("needs_review") == 1


def test_the_german_profile_goes_through_the_same_checks(monkeypatch):
    german = [{"front": "der Tisch", "back": "table<br>Plural: die Tische", "type": "basic", "tags": ["vocabulary"], "t": 1}]
    use(monkeypatch, lambda p: returned(p, {0: {"back": "table<br>Plural: die Tisches", "fixed_fact": True, "note": "plural of Tisch is Tische"}}))
    out, res = quality.improve_cards(german, "Deutsch", api_key="k", use_pro=False, profile="german")
    assert out[0]["front"] == "der Tisch" and "needs_review" in out[0]["tags"] and res.needs_review == 1


# ── the report ──────────────────────────────────────────────────────────────────

def test_the_report_lists_what_needs_review_first_with_before_and_after(monkeypatch):
    _, res, _ = run(monkeypatch)
    md = integrity.render_changes_report("Neuro", res.changes, total_cards=5)
    assert md.startswith("# Quality-check changes — Neuro")
    assert "4 of 5 cards were changed" in md and "2 need your review" in md
    review, other = md.split("## Needs review")[1].split("## Other changes")
    assert "150 mL" in review and "100 mL" in review and "villi reabsorb CSF" in review
    assert "numbers" in review.lower()
    assert "diaphragm muscle" in other and "diaphragm muscle" not in review


def test_no_changes_no_report():
    assert integrity.render_changes_report("D", [], total_cards=5) == ""
