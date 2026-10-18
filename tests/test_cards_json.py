"""cards_json.parse_cards — tolerant parsing of LLM card output."""
import json

import pytest

from cards_json import parse_cards, parse_cards_ex


def card(i, **kw):
    return {"front": f"Q{i}", "back": f"A{i}", "type": "basic", "tags": [], **kw}


def payload(n):
    return json.dumps({"cards": [card(i) for i in range(1, n + 1)]})


def test_clean_json():
    assert len(parse_cards(payload(3))) == 3


def test_markdown_fences():
    assert len(parse_cards("```json\n" + payload(2) + "\n```")) == 2


def test_trailing_prose_with_braces_does_not_lose_the_chunk():
    raw = payload(2) + "\n\nNote: I used the {c1::cloze} rules as asked."
    assert len(parse_cards(raw)) == 2


def test_leading_prose_with_braces():
    raw = "Sure! Here is the {requested} output:\n" + payload(2)
    assert len(parse_cards(raw)) == 2


def test_cloze_markup_inside_strings_is_fine():
    raw = json.dumps({"cards": [card(1, front="What makes CSF?<br><br>{{c1::Choroid plexus}}", type="cloze")]})
    cards = parse_cards(raw + "\nDone {{c9::}}")
    assert cards[0]["front"].endswith("{{c1::Choroid plexus}}")


def test_truncated_reply_keeps_the_complete_cards():
    full = payload(4)
    cut = full[: full.index('"Q4"') + 12]          # 4th card cut off mid-object
    cards, salvaged = parse_cards_ex(cut)
    assert salvaged is True
    assert [c["front"] for c in cards] == ["Q1", "Q2", "Q3"]


def test_truncated_inside_a_string_drops_only_that_card():
    full = payload(3)
    cut = full[: full.index('"A3"') + 3]
    assert [c["front"] for c in parse_cards(cut)] == ["Q1", "Q2"]


def test_top_level_list_is_accepted():
    assert len(parse_cards(json.dumps([card(1), card(2)]))) == 2


def test_cyrillic_and_german_text():
    raw = json.dumps({"cards": [card(1, front="Қайси нерв диафрагмани таъминлайди?"),
                                card(2, front="der Arzt / die Ärztin")]}, ensure_ascii=False)
    assert [c["front"] for c in parse_cards(raw)][1] == "der Arzt / die Ärztin"


def test_items_without_a_front_are_dropped():
    raw = json.dumps({"cards": [card(1), {"back": "orphan"}, "junk", card(2)]})
    assert [c["front"] for c in parse_cards(raw)] == ["Q1", "Q2"]


@pytest.mark.parametrize("raw", ["", "no json here at all", "{}", '{"cards": []}', '{"other": [1,2]}'])
def test_nothing_usable_returns_empty_list(raw):
    assert parse_cards(raw) == []


def test_non_string_raises_so_callers_fall_back_to_next_provider():
    with pytest.raises(TypeError):
        parse_cards(None)


# ── wiring: the two modules that used to carry their own greedy-regex copy ──

def test_anki_gen_and_generator_use_the_tolerant_parser():
    import anki_gen
    import generator

    prose = payload(2) + "\nNote: {see above}"
    cut = payload(4)[: payload(4).index('"Q4"') + 12]
    for mod in (anki_gen, generator):
        assert len(mod._parse_response(prose)) == 2, mod.__name__
        assert len(mod._parse_response(cut)) == 3, mod.__name__
