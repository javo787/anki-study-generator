"""
The whole 'no silent edits' chain, with the REAL filter / quality / format code and fake models:
what the AI changed is tagged, listed in a report and warned about; what it must not change is
protected; the source-link time survives without any repair.
"""
import csv
import json
import os
import re
import types

import pytest

import filter as flt
import format_pass as fp
import generator
import pipeline
import quality
import sources
import usage
from fakes import FakeGemini, Sleeper, block_entries, gemini_client

ID = "dQw4w9WgXcQ"


class FakeGroq:
    """Plays filter, duplicate-check and format models; the test sets what each answers."""
    def __init__(self):
        self.filter_reply = lambda payload: {"removed": []}
        self.format_reply = lambda payload: [dict(p) for p in payload]
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model, messages, temperature):
        system, user = messages[0]["content"], messages[1]["content"]
        if system == flt.FILTER_SYSTEM:
            out = self.filter_reply(json.loads(user[user.index("Cards:\n") + 7: user.rindex("\n\nReturn")]))
        elif system == flt.DEDUP_SYSTEM:
            out = {"results": []}
        elif system == fp.FORMAT_SYSTEM:
            out = {"cards": self.format_reply(json.loads(user[user.index("Cards:\n") + 7: user.rindex("\n\nReturn")]))}
        else:
            raise AssertionError("unexpected system prompt")
        msg = types.SimpleNamespace(content=out if isinstance(out, str) else json.dumps(out))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeQualityModel:
    def __init__(self):
        self.reply = lambda payload: [dict(p, improved=False, mnemonic_added=False) for p in payload]
        self.models = self

    def generate_content(self, model, contents, config=None):
        payload = json.loads(contents[contents.index("Cards:\n") + 7: contents.rindex("\n\nReturn")])
        out = self.reply(payload)
        return types.SimpleNamespace(text=out if isinstance(out, str) else json.dumps({"cards": out}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    s, real_time = Sleeper(), __import__("time").time
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(usage, "USAGE_FILE", str(tmp_path / "usage.json"))
    for mod in (pipeline, generator):
        monkeypatch.setattr(mod, "time", types.SimpleNamespace(sleep=s, time=real_time))
    for mod in (quality, flt, fp):
        monkeypatch.setattr(mod.time, "sleep", lambda x: None)
    monkeypatch.setattr(generator, "get_transcript_entries", lambda *a, **k: block_entries(n_blocks=3))
    monkeypatch.setattr(pipeline, "load_config", lambda: {"gemini_api_key": "k", "gemini_pro_key": "p", "groq_api_key": "g"})
    monkeypatch.setattr(pipeline, "build_clients", lambda c: [gemini_client(FakeGemini())])
    monkeypatch.setattr(pipeline, "restore_times", lambda final, raw: 0)          # prove the time survives WITHOUT the repair
    groq, qm = FakeGroq(), FakeQualityModel()
    monkeypatch.setattr(flt, "Groq", lambda api_key: groq)
    monkeypatch.setattr(fp, "Groq", lambda api_key: groq)
    monkeypatch.setattr(quality.genai, "Client", lambda api_key: qm)
    return types.SimpleNamespace(groq=groq, quality=qm, out=tmp_path / "out")


def run():
    return pipeline.run(pipeline.PipelineConfig(source="youtube", input=f"https://youtu.be/{ID}", deck="Neuro"))


def rows(res):
    with open(res.filename, encoding="utf-8", newline="") as f:
        return list(csv.reader([l for l in f.read().splitlines(True) if not l.startswith("#")], delimiter="\t"))


def edit(payload):
    """Quality model: a mnemonic on card 0 (harmless), a changed number on 1, a corrected fact on 2."""
    out = [dict(p, improved=False, mnemonic_added=False) for p in payload]
    out[0].update(back=out[0]["back"] + "<br>💡 remember it well", improved=True, mnemonic_added=True)
    out[1].update(back=out[1]["back"] + " 150")
    out[2].update(back="completely different words here", fixed_fact=True, note="was wrong")
    return out


def test_changes_are_tagged_reported_and_warned_about_and_the_links_survive(env):
    env.quality.reply = edit
    res = run()

    tagged = [r for r in rows(res) if "needs_review" in r[3].split()]
    assert len(tagged) == 2                                                       # the number change and the corrected fact
    assert len(rows(res)) == 9 and all("?t=" in r[2] for r in rows(res))          # every card still links to its moment

    assert res.report and os.path.exists(res.report) and res.report.endswith("_changes.md")
    md = open(res.report, encoding="utf-8").read()
    assert "3 of 9 cards were changed" in md and "2 need your review" in md and "numbers changed" in md and "was wrong" in md

    q = res.stages["quality"]
    assert q["changed"] == 3 and q["needs_review"] == 2 and q["mnemonics_added"] == 1 and q["report"] == os.path.basename(res.report)
    assert any("2 card(s)" in e and "needs_review" in e and os.path.basename(res.report) in e for e in res.errors)


def test_a_quality_pass_that_changes_nothing_leaves_no_report(env):
    res = run()
    assert res.report == "" and not any(f.endswith("_changes.md") for f in os.listdir(env.out))
    assert res.errors == [] and res.stages["quality"]["changed"] == 0


def test_an_answer_that_cannot_be_matched_is_reported_and_changes_nothing(env):
    env.quality.reply = lambda payload: "I cannot help with that"
    res = run()
    assert len(rows(res)) == 9 and res.stages["quality"]["changed"] == 0
    assert res.stages["quality"]["rejected_batches"] == 1 and any("could not be matched" in e for e in res.errors)


def test_a_mass_removal_by_the_filter_is_ignored_and_reported(env):
    env.groq.filter_reply = lambda payload: {"removed": [{"id": i, "reason": "trivial"} for i in range(len(payload) - 1)]}
    res = run()
    assert len(rows(res)) == 9
    assert any("implausible" in e for e in res.errors)


def test_the_format_pass_cannot_change_what_a_card_says(env):
    original = {}

    def reword(payload):
        original.update({p["id"]: p["back"] for p in payload})
        out = [dict(p) for p in payload]
        out[0]["back"] = "an entirely invented answer"
        return out

    env.groq.format_reply = reword
    res = run()
    assert original, "the format model was never asked"                          # the test is not vacuous
    assert not any("invented" in r[2] for r in rows(res)) and len(rows(res)) == 9


def test_server_returns_the_report_name(monkeypatch):
    import server

    def fake_run(**kw):
        return types.SimpleNamespace(total_final=1, total_raw=1, total_filtered=1, filename="out/x.txt", report="out/x_changes.md",
                                     duration_sec=0.1, stages={}, cards=[], errors=[])

    monkeypatch.setattr(server, "run_youtube", fake_run)
    r = server.app.test_client().post("/api/generate/youtube", json={"youtube_url": f"https://youtu.be/{ID}"})
    assert r.get_json()["report"] == "x_changes.md"


def test_the_review_tag_reaches_real_anki(env, tmp_path):
    pytest.importorskip("anki")
    from anki.collection import Collection
    from anki.import_export_pb2 import ImportCsvRequest
    env.quality.reply = edit
    res = run()
    path = os.path.abspath(res.filename)
    col = Collection(str(tmp_path / "c.anki2"))
    try:
        col.import_csv(ImportCsvRequest(path=path, metadata=col.get_csv_metadata(path, None)))
        assert len(col.find_notes("")) == 9
        flagged = col.find_notes("tag:needs_review")                    # what a learner would type in Anki's browser
        assert len(flagged) == 2
        assert all("?t=" in col.get_note(n).fields[1] for n in col.find_notes(""))
    finally:
        col.close()
