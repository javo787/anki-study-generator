"""
Export format tests for pipeline.save_cards().

These run against the REAL Anki import engine when the `anki` package is
installed (pip install -r requirements-dev.txt) — that is the only honest way
to know a card file imports the way we intend.
"""
import csv
import os

import pytest

import pipeline

CARDS = [
    {"front": "Which nerve supplies the diaphragm?", "back": "• Phrenic nerve\n💡 C3,4,5",
     "type": "basic", "tags": ["neuroanatomy"]},
    {"front": '"Dura" is the outermost meningeal layer. True or false?', "back": 'He said "yes"',
     "type": "basic", "tags": ["meninges"]},
    {"front": "What produces CSF?<br><br>{{c1::Choroid plexus}}", "back": "Located in ventricles",
     "type": "cloze", "tags": ["csf"]},
    {"front": "The {{c1::arachnoid villi}} reabsorb CSF.", "back": "into the venous sinuses",
     "type": "cloze", "tags": ["csf"]},
]


@pytest.fixture
def saved_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path))
    return pipeline.save_cards(CARDS, "Test Deck")


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        body = [ln for ln in f.read().splitlines(True) if not ln.startswith("#")]
    return list(csv.reader(body, delimiter="\t"))


def test_single_header_block_only(saved_file):
    """Anki ignores '#' lines after the first data row, so a second
    '#notetype:Cloze' header block mid-file silently does nothing."""
    lines = open(saved_file, encoding="utf-8").read().splitlines()
    first_data = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
    assert not any(ln.startswith("#") for ln in lines[first_data:]), \
        "header lines after the first data row are ignored by Anki"


def test_notetype_comes_from_a_column(saved_file):
    text = open(saved_file, encoding="utf-8").read()
    assert "#notetype column:1" in text
    rows = _rows(saved_file)
    assert [r[0] for r in rows] == ["Basic", "Basic", "Cloze", "Cloze"]


def test_quotes_and_newlines_survive_round_trip(saved_file):
    rows = _rows(saved_file)
    assert rows[1][1] == '"Dura" is the outermost meningeal layer. True or false?'
    assert rows[1][2] == 'He said "yes"'
    assert "\n" not in rows[0][2] and "<br>" in rows[0][2]


def test_unknown_or_capitalised_type_is_not_silently_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path))
    cards = [{"front": "Q", "back": "A", "type": "Basic", "tags": []},
             {"front": "Q2 {{c1::x}}", "back": "", "type": "CLOZE", "tags": []},
             {"front": "Q3", "back": "A3", "tags": []}]          # no type at all
    rows = _rows(pipeline.save_cards(cards, "D"))
    assert [r[0] for r in rows] == ["Basic", "Cloze", "Basic"]


def test_tags_with_spaces_become_single_tags(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path))
    rows = _rows(pipeline.save_cards(
        [{"front": "Q", "back": "A", "type": "basic", "tags": ["CSF circulation", "x"]}], "D"))
    assert rows[0][3] == "CSF_circulation x"


def test_imports_into_real_anki(saved_file, tmp_path):
    """Golden test: what lands in Anki is what we meant to export."""
    pytest.importorskip("anki")
    from anki.collection import Collection
    try:
        from anki.import_export_pb2 import ImportCsvRequest
    except ImportError:                                  # pragma: no cover
        pytest.skip("this anki version has no CSV import API")

    col = Collection(str(tmp_path / "collection.anki2"))
    try:
        meta = col.get_csv_metadata(os.path.abspath(saved_file), None)
        log = col.import_csv(ImportCsvRequest(path=os.path.abspath(saved_file), metadata=meta))
        assert len(log.log.new) == 4

        notes = [col.get_note(n) for n in sorted(col.find_notes(""))]
        by_type = [(n.note_type()["name"], len(n.card_ids())) for n in notes]
        assert by_type == [("Basic", 1), ("Basic", 1), ("Cloze", 1), ("Cloze", 1)]
        assert notes[1].fields[0] == '"Dura" is the outermost meningeal layer. True or false?'
        assert notes[2].fields[0] == "What produces CSF?<br><br>{{c1::Choroid plexus}}"
    finally:
        col.close()
