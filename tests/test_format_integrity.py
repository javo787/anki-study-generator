"""The format pass fixes FORMAT. It must not change what a card says — and must never lose a card."""
import json
import types

import pytest

import format_pass as fp


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(fp.time, "sleep", lambda s: None)


class Groq:
    def __init__(self, script):
        self.script, self.payloads = script, []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model, messages, temperature):
        user = messages[1]["content"]
        payload = json.loads(user[user.index("Cards:\n") + 7: user.rindex("\n\nReturn")])
        self.payloads.append(payload)
        out = self.script(payload)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps({"cards": out})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])


def deck():
    mk = lambda f, b, ty="basic": {"front": f, "back": b, "type": ty, "tags": ["neuro"], "t": 7}
    return [mk("What produces CSF?", "**Choroid plexus**\n• in the ventricles"),
            mk("Normal CSF volume in adults?", "150 mL"),
            mk("The ___ drains CSF. (arachnoid villi)", "into the sinuses", "cloze"),
            mk("Which nerve supplies the diaphragm?", "Phrenic nerve")]


def run(monkeypatch, script):
    fake = Groq(script)
    monkeypatch.setattr(fp, "Groq", lambda api_key: fake)
    out, res = fp.format_cards(deck(), "Neuro", "key")
    return out, res, fake


def echo(payload, edits=None):
    out = [dict(p) for p in payload]
    for i, e in (edits or {}).items():
        out[i].update(e)
    return out


def test_cards_are_sent_with_ids_and_only_what_a_format_pass_needs(monkeypatch):
    _, _, fake = run(monkeypatch, echo)
    p = fake.payloads[0]
    assert [x["id"] for x in p] == [0, 1, 2, 3] and set(p[0]) == {"id", "front", "back", "type"}


def test_a_formatting_change_is_accepted(monkeypatch):
    out, res, _ = run(monkeypatch, lambda p: echo(p, {0: {"back": "<b>Choroid plexus</b><br>• in the ventricles"}}))
    assert out[0]["back"] == "<b>Choroid plexus</b><br>• in the ventricles" and res.ai_content_rejected == 0


def test_a_content_change_is_reverted_to_the_original(monkeypatch):
    out, res, _ = run(monkeypatch, lambda p: echo(p, {0: {"back": "Arachnoid villi at the sagittal sinus"}}))
    assert "Choroid plexus" in out[0]["back"] and res.ai_content_rejected == 1


def test_a_changed_number_is_reverted_even_if_the_words_are_the_same(monkeypatch):
    out, res, _ = run(monkeypatch, lambda p: echo(p, {1: {"back": "100 mL"}}))
    assert out[1]["back"] == "150 mL" and res.ai_content_rejected == 1


def test_a_card_the_model_forgot_is_kept(monkeypatch):
    out, res, _ = run(monkeypatch, lambda p: echo(p)[:2])
    assert len(out) == 4 and res.cards_out == 4 and res.ai_missing == 2
    assert out[3]["front"] == "Which nerve supplies the diaphragm?"


def test_type_tags_and_the_source_time_cannot_be_changed(monkeypatch):
    out, _, _ = run(monkeypatch, lambda p: echo(p, {0: {"type": "cloze", "tags": ["evil"], "t": 999, "back": "<b>Choroid plexus</b><br>• in the ventricles"}}))
    assert out[0]["type"] == "basic" and out[0]["tags"] == ["neuro"] and all(c["t"] == 7 for c in out)


@pytest.mark.parametrize("junk", ["sorry", "{}", '{"cards": []}', RuntimeError("500")])
def test_unusable_answers_keep_every_card_and_the_local_repairs_still_run(monkeypatch, junk):
    out, res, _ = run(monkeypatch, lambda p: junk)
    assert len(out) == 4 and out[3]["front"] == "Which nerve supplies the diaphragm?"
    assert "{{c1::arachnoid villi}}" in out[2]["front"] and "___" not in out[2]["front"]      # the deterministic repair is the safety net
    assert "<b>Choroid plexus</b>" in out[0]["back"]                                        # markdown -> HTML locally


def test_without_ids_the_same_number_of_cards_is_matched_by_position(monkeypatch):
    out, _, _ = run(monkeypatch, lambda p: [{k: v for k, v in c.items() if k != "id"} for c in echo(p, {0: {"back": "<b>Choroid plexus</b><br>• in the ventricles"}})])
    assert out[0]["back"].startswith("<b>Choroid")


def test_the_prompt_asks_for_ids_back():
    assert '"id"' in fp.FORMAT_SYSTEM and "id" in fp.FORMAT_SYSTEM.lower().split("output")[0]


def test_the_summary_mentions_what_was_reverted(monkeypatch):
    _, res, _ = run(monkeypatch, lambda p: echo(p, {0: {"back": "something else entirely"}})[:3])
    assert "reverted" in res.summary() and res.ai_missing == 1
