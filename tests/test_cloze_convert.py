"""pipeline._convert_to_anki_cloze must only touch legacy '___ (answer)' cards."""
import pipeline


def test_legacy_blank_is_converted():
    assert pipeline._convert_to_anki_cloze("The ___ drains CSF. (Arachnoid villi)") == \
        "The {{c1::Arachnoid villi}} drains CSF."


def test_valid_cloze_with_trailing_parenthetical_is_left_alone():
    text = "What produces CSF?<br><br>{{c1::Choroid plexus}} (in the ventricles)"
    assert pipeline._convert_to_anki_cloze(text) == text


def test_valid_inline_cloze_with_trailing_parenthetical_is_left_alone():
    text = "The {{c1::dura mater}} is the outermost layer (of three)"
    assert pipeline._convert_to_anki_cloze(text) == text
