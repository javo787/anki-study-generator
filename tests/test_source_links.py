"""
Source links: every card from a YouTube video gets a link to the moment it was said.
All offline — see tests/fakes.py.
"""
import csv
import os
import types

import pytest

import chunker
import generator
import pipeline
import sources
import usage
from chunker import Chunk
from fakes import (quality_stub, FakeGemini, Sleeper, block_entries, gemini_client, pseudo, word_entries)

ID = "dQw4w9WgXcQ"


def make_chunk(entries, start=None, end=None, video_id=ID):
    return Chunk(text=" ".join(e["text"] for e in entries), index=1, total=1, source="youtube",
                 start_time=entries[0]["start"] if start is None else start,
                 end_time=entries[-1]["start"] + entries[-1]["duration"] if end is None else end,
                 entries=entries, video_id=video_id)


def card_from(entries, *idx, front_extra="Which structure"):
    words = " ".join(entries[i]["text"] for i in idx).split()
    return {"front": f"{front_extra} {' '.join(words[:3])}?", "back": " ".join(words[3:]),
            "type": "basic", "tags": []}


# ── formatting ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sec, label", [(0, "00:00"), (303, "05:03"), (303.9, "05:03"), (3599, "59:59"),
                                        (3725, "1:02:05"), (36000, "10:00:00")])
def test_format_time(sec, label):
    assert sources.format_time(sec) == label


def test_video_url_uses_whole_seconds():
    assert sources.video_url(ID, 303.7) == f"https://youtu.be/{ID}?t=303"


# ── locating a card inside its chunk ────────────────────────────────────────────

def test_card_is_located_at_the_caption_it_was_taken_from():
    es = word_entries(120)
    chunk = make_chunk(es)
    cards = [card_from(es, 17), card_from(es, 63), card_from(es, 100)]
    sources.locate_cards(cards, chunk)
    # start of the caption minus the 5 s lead-in
    assert [c["t"] for c in cards] == [int(es[17]["start"]) - 5, int(es[63]["start"]) - 5, int(es[100]["start"]) - 5]


def test_a_fact_spanning_two_captions_points_at_the_first_one():
    es = word_entries(120)
    cards = [card_from(es, 40, 41)]
    sources.locate_cards(cards, make_chunk(es))
    assert cards[0]["t"] == int(es[40]["start"]) - 5


def test_lead_in_never_goes_before_zero():
    es = word_entries(30)
    cards = [card_from(es, 0)]
    sources.locate_cards(cards, make_chunk(es))
    assert cards[0]["t"] == 0


@pytest.mark.parametrize("front, back", [
    ("Totally unrelated question?", "nothing from the lecture here"),       # no overlap
    ("Қайси нерв диафрагмани таъминлайди?", "Френик нерв"),                  # other language/script
    ("", ""),
])
def test_without_a_confident_match_the_link_points_to_the_start_of_the_chunk(front, back):
    es = word_entries(60, t0=300.0)
    cards = [{"front": front, "back": back, "type": "basic", "tags": []}]
    sources.locate_cards(cards, make_chunk(es))
    assert cards[0]["t"] == 300


def test_one_shared_word_is_not_enough_to_claim_a_position():
    es = word_entries(60, t0=300.0)
    w = es[30]["text"].split()[0]
    cards = [{"front": f"What about {w}?", "back": "something else entirely", "type": "basic", "tags": []}]
    sources.locate_cards(cards, make_chunk(es))
    assert cards[0]["t"] == 300


def test_words_that_occur_everywhere_carry_no_signal():
    es = word_entries(60, t0=300.0)
    for e in es:                                      # every caption says these
        e["text"] += " structure drains fluid"
    cards = [{"front": "Which structure drains fluid?", "back": "structure fluid drains", "type": "basic", "tags": []}]
    sources.locate_cards(cards, make_chunk(es))
    assert cards[0]["t"] == 300


def test_chunk_without_entries_falls_back_to_its_start():
    chunk = Chunk(text="x", index=1, total=1, source="youtube", start_time=120, end_time=300, video_id=ID)
    cards = [{"front": "q", "back": "a", "type": "basic", "tags": []}]
    sources.locate_cards(cards, chunk)
    assert cards[0]["t"] == 120


# ── getting the time back after the LLM passes rewrote the cards ────────────────

def _raw_cards():
    return [{"front": f"What does {pseudo(10 + i)} do in the {pseudo(60 + i)}?", "back": f"{pseudo(110 + i)} {pseudo(160 + i)} {pseudo(210 + i)}",
             "type": "basic", "tags": [], "t": 100 * (i + 1)} for i in range(4)]


def test_rewritten_cards_get_their_time_back_from_the_matching_original():
    raw = _raw_cards()
    rewritten = [{"front": "Better: " + c["front"], "back": c["back"] + " 💡 mnemonic to remember it", "type": "basic", "tags": []}
                 for c in reversed(raw)]                                   # also reordered
    found = sources.restore_times(rewritten, raw)
    assert found == 4
    assert [c["t"] for c in rewritten] == [400, 300, 200, 100]


def test_a_valid_time_is_kept_and_an_invalid_one_is_replaced():
    raw = _raw_cards()
    keep = dict(raw[0], t=555)                       # model kept it (even if different): trust a valid int
    bad = {k: v for k, v in raw[1].items()}
    bad["t"] = "05:03"                               # model mangled it
    sources.restore_times([keep, bad], raw)
    assert keep["t"] == 555 and bad["t"] == 200


def test_cards_that_match_nothing_get_no_time():
    raw = _raw_cards()
    stranger = {"front": "Completely new card?", "back": "no relation", "type": "basic", "tags": []}
    assert sources.restore_times([stranger], raw) == 0
    assert "t" not in stranger


# ── chunks carry their captions and the video id ────────────────────────────────

def test_youtube_chunks_keep_their_captions_including_a_merged_tail():
    body = block_entries(n_blocks=1, block_seconds=590)
    end = body[-1]["start"] + body[-1]["duration"]
    outro = [{"text": "thank you for watching", "start": end + 4.0, "duration": 4.0}]
    chunks = chunker.chunk_youtube(body + outro)
    assert len(chunks) == 1 and chunks[0].entries == body + outro


def test_generate_from_youtube_stamps_the_video_id_on_every_chunk(monkeypatch):
    monkeypatch.setattr(generator, "get_transcript_entries", lambda *a, **k: block_entries(n_blocks=3))
    seen = []
    monkeypatch.setattr(generator, "_run_pipeline", lambda chunks, *a, **k: seen.extend(chunks) or generator.GenerationResult([], 0, 0))
    generator.generate_from_youtube(f"https://youtu.be/{ID}", "D", [])
    assert len(seen) == 3 and {c.video_id for c in seen} == {ID}


def test_generation_locates_cards_of_youtube_chunks_only(monkeypatch):
    monkeypatch.setattr(generator, "time", types.SimpleNamespace(sleep=Sleeper()))
    es = word_entries(80)
    yt = make_chunk(es)
    pdf = Chunk(text="x", index=1, total=1, source="pdf", page_from=1, page_to=2)

    class Fixed(FakeGemini):
        def generate_content(self, model, contents, config=None):
            if "main anatomical or medical topic" in contents:
                return super().generate_content(model, contents, config)
            import json
            c = card_from(es, 33)
            return type("R", (), {"text": json.dumps({"cards": [c]})})()

    res = generator._run_pipeline([yt], "D", [gemini_client(Fixed())], False, None, "en")
    assert res.cards[0]["t"] == int(es[33]["start"]) - 5
    res = generator._run_pipeline([pdf], "D", [gemini_client(Fixed())], False, None, "en")
    assert "t" not in res.cards[0]


# ── export ──────────────────────────────────────────────────────────────────────

def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        body = [ln for ln in f.read().splitlines(True) if not ln.startswith("#")]
    return list(csv.reader(body, delimiter="\t"))


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path))
    return tmp_path


def test_save_cards_appends_the_link_to_the_back(out):
    cards = [{"front": "Q", "back": "• A\n• B", "type": "basic", "tags": [], "t": 303},
             {"front": "Cloze {{c1::x}}", "back": "", "type": "cloze", "tags": [], "t": 3725},
             {"front": "No time", "back": "A", "type": "basic", "tags": []},
             {"front": "Bad time", "back": "A", "type": "basic", "tags": [], "t": "05:03"},
             {"front": "Negative", "back": "A", "type": "basic", "tags": [], "t": -4}]
    rows = _rows(pipeline.save_cards(cards, "D", video_id=ID))
    assert rows[0][2] == f'• A<br>• B<br><br><small>▶ <a href="https://youtu.be/{ID}?t=303">05:03</a></small>'
    assert rows[1][2] == f'<small>▶ <a href="https://youtu.be/{ID}?t=3725">1:02:05</a></small>'    # empty back: no leading <br>
    assert rows[2][2] == "A" and rows[3][2] == "A" and rows[4][2] == "A"


def test_save_cards_without_a_video_id_adds_nothing(out):
    rows = _rows(pipeline.save_cards([{"front": "Q", "back": "A", "type": "basic", "tags": [], "t": 303}], "D"))
    assert rows[0][2] == "A"


def test_link_survives_a_real_anki_import(out, tmp_path):
    pytest.importorskip("anki")
    from anki.collection import Collection
    from anki.import_export_pb2 import ImportCsvRequest
    path = os.path.abspath(pipeline.save_cards(
        [{"front": "Q", "back": "A", "type": "basic", "tags": [], "t": 303}], "D", video_id=ID))
    col = Collection(str(tmp_path / "c.anki2"))
    try:
        col.import_csv(ImportCsvRequest(path=path, metadata=col.get_csv_metadata(path, None)))
        note = col.get_note(col.find_notes("")[0])
        assert f'<a href="https://youtu.be/{ID}?t=303">05:03</a>' in note.fields[1]
    finally:
        col.close()


# ── the whole pipeline ──────────────────────────────────────────────────────────

@pytest.fixture
def env(out, tmp_path, monkeypatch):
    s = Sleeper()
    real_time = __import__("time").time
    monkeypatch.setattr(usage, "USAGE_FILE", str(tmp_path / "usage.json"))
    monkeypatch.setattr(pipeline, "time", types.SimpleNamespace(sleep=s, time=real_time))
    monkeypatch.setattr(generator, "time", types.SimpleNamespace(sleep=s, time=real_time))
    monkeypatch.setattr(generator, "get_transcript_entries", lambda *a, **k: block_entries(n_blocks=3))
    monkeypatch.setattr(pipeline, "build_proxy_config", lambda cfg: None)
    monkeypatch.setattr(pipeline, "load_config", lambda: {"gemini_api_key": "k", "gemini_pro_key": "p"})
    monkeypatch.setattr(pipeline, "build_clients", lambda c: [gemini_client(FakeGemini())])
    return out


def _final_rows(res):
    return _rows(res.filename)


def _t_values(rows):
    import re
    return [int(m.group(1)) for r in rows for m in [re.search(r"\?t=(\d+)", r[2])] if m]


def test_pipeline_adds_links_to_every_card_of_a_youtube_run(env, monkeypatch):
    monkeypatch.setattr(pipeline, "improve_cards", lambda **kw: quality_stub(kw["cards"]))
    res = pipeline.run(pipeline.PipelineConfig(source="youtube", input=f"https://youtu.be/{ID}", deck="D",
                                               run_filter=False, run_quality=True))
    rows = _final_rows(res)
    assert len(rows) == 9 and len(_t_values(rows)) == 9
    assert sorted(set(_t_values(rows))) == [0, 303, 606]            # one per chunk (cards match no caption)


def test_links_survive_quality_rewriting_that_drops_unknown_fields(env, monkeypatch):
    def rewrite(**kw):                                              # like Gemini Pro: new wording, no extra keys
        return quality_stub([{"front": "Better: " + c["front"], "back": c["back"] + " 💡 remember", "type": c["type"], "tags": c["tags"]}
                             for c in kw["cards"]])
    monkeypatch.setattr(pipeline, "improve_cards", rewrite)
    res = pipeline.run(pipeline.PipelineConfig(source="youtube", input=f"https://youtu.be/{ID}", deck="D",
                                               run_filter=False, run_quality=True))
    ts = _t_values(_final_rows(res))
    assert len(ts) == 9
    assert {t: ts.count(t) for t in set(ts)} == {0: 3, 303: 3, 606: 3}   # each chunk's 3 cards kept their own time


def test_source_links_can_be_switched_off(env):
    res = pipeline.run(pipeline.PipelineConfig(source="youtube", input=f"https://youtu.be/{ID}", deck="D",
                                               run_filter=False, run_quality=False, source_links=False))
    assert _t_values(_final_rows(res)) == []


def test_raw_prefilter_file_has_links_too(env):
    pipeline.run(pipeline.PipelineConfig(source="youtube", input=f"https://youtu.be/{ID}", deck="D",
                                         run_filter=False, run_quality=False))
    raw = [f for f in os.listdir(pipeline.OUTPUT_DIR) if "RAW_prefilter" in f]
    assert raw and _t_values(_rows(os.path.join(pipeline.OUTPUT_DIR, raw[0])))


# ── server ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("body, expected", [({}, True), ({"source_links": False}, False), ({"source_links": True}, True)])
def test_server_forwards_the_source_links_option(monkeypatch, body, expected):
    import server
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return types.SimpleNamespace(total_final=0, total_raw=0, total_filtered=0, filename="x.txt", duration_sec=0.1,
                                     stages={}, cards=[], errors=[])

    monkeypatch.setattr(server, "run_youtube", fake_run)
    r = server.app.test_client().post("/api/generate/youtube", json={"youtube_url": f"https://youtu.be/{ID}", **body})
    assert r.status_code == 200 and seen["source_links"] is expected


# ── long captions (Whisper sentences) share everyday words with their neighbours ─

def test_long_captions_that_share_common_words_do_not_pull_the_link_to_a_neighbour():
    """Whisper gives whole sentences (10-20 s). A card taken from one of them also shares a few
    everyday words with the sentences before it; the link must still point at ITS sentence."""
    common = ["pressure", "fluid", "which", "csf", "structure"]
    entries = []
    for k in range(60):
        unique = [pseudo(k * 12 + j) for j in range(10)]
        mixed = unique[:4] + [common[(k + j) % 5] for j in range(3)] + unique[4:]
        entries.append({"text": " ".join(mixed), "start": 100.0 + k * 15.0, "duration": 14.0})
    chunk = make_chunk(entries)
    cards = [{"front": " ".join(entries[k]["text"].split()[:4]) + "?", "back": " ".join(entries[k]["text"].split()[4:]),
              "type": "basic", "tags": []} for k in (7, 23, 41, 58)]
    sources.locate_cards(cards, chunk)
    assert [c["t"] for c in cards] == [int(entries[k]["start"]) - 5 for k in (7, 23, 41, 58)]


def test_a_fact_spread_over_two_long_captions_still_points_at_the_first_when_both_contribute():
    entries = [{"text": " ".join(pseudo(k * 6 + j) for j in range(6)), "start": 10.0 * k, "duration": 9.0} for k in range(40)]
    card = {"front": "x?", "back": " ".join(entries[20]["text"].split()[3:] + entries[21]["text"].split()[:3]),
            "type": "basic", "tags": []}
    sources.locate_cards([card], make_chunk(entries))
    assert card["t"] == int(entries[20]["start"]) - 5


# ── card boilerplate must not outvote a short card's real content ───────────────

def test_boilerplate_is_what_most_cards_of_a_chunk_share():
    cards = [{"front": f"Which structure {pseudo(i)}?", "back": f"{pseudo(100 + i)} remember", "type": "basic", "tags": []} for i in range(8)]
    from textmatch import card_text, content_tokens
    bp = sources._boilerplate([content_tokens(card_text(c)) for c in cards])
    assert {"which", "struc", "remem"} <= bp and not any(pseudo(i)[:5] in bp for i in range(8))


def test_with_few_cards_nothing_is_called_boilerplate():
    from textmatch import content_tokens
    assert sources._boilerplate([content_tokens("which structure remember")] * (sources.BOILERPLATE_MIN_CARDS - 1)) == set()


def test_a_short_card_is_not_dragged_to_an_unrelated_sentence_that_happens_to_use_its_wording():
    entries = word_entries(60, step=6.0, words_per_entry=5)
    entries[5]["text"] += " which structure remember"          # an unrelated sentence that uses the cards' wording once
    chunk = make_chunk(entries)
    targets = [10, 20, 30, 40, 50, 15, 25, 35]
    cards = []
    for k in targets:                                           # every card is worded "Which structure ... remember"
        w = entries[k]["text"].split()
        cards.append({"front": f"Which structure {w[0]} {w[1]}?", "back": f"{w[2]} remember", "type": "basic", "tags": []})
    sources.locate_cards(cards, chunk)
    assert [c["t"] for c in cards] == [int(entries[k]["start"]) - 5 for k in targets]


def test_the_same_short_card_located_alone_still_uses_all_its_words():
    """(documents the batch behaviour: boilerplate is only judged across a chunk's cards)"""
    entries = word_entries(60, step=6.0, words_per_entry=5)
    chunk = make_chunk(entries)
    w = entries[30]["text"].split()
    card = {"front": f"Which structure {w[0]} {w[1]}?", "back": f"{w[2]}", "type": "basic", "tags": []}
    sources.locate_cards([card], chunk)
    assert card["t"] == int(entries[30]["start"]) - 5
