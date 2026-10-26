"""
Every prompt template that is filled with str.format() must actually format.
(quality.QUALITY_SYSTEM did not: its JSON example used single braces, so the whole
Gemini quality step raised KeyError on its first batch — on every run.)
"""
import types

import pytest

import anki_gen
import filter as flt
import generator
import quality

KW = dict(chunk_index=1, chunk_total=3, topic="CSF", language_instruction="Write in English.",
          count=2, deck="D", cards_json="[]")

TEMPLATES = {
    "anki_gen.SYSTEM_PROMPT":     anki_gen.SYSTEM_PROMPT,
    "generator.GROQ_CARD_PROMPT": generator.GROQ_CARD_PROMPT,
    "quality.QUALITY_SYSTEM":     quality.QUALITY_SYSTEM,
    "filter.FILTER_USER":         flt.FILTER_USER,
}


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_formats_and_leaves_no_placeholders(name):
    out = TEMPLATES[name].format(**KW)
    for key in ("{language_instruction}", "{chunk_index}", "{topic}", "{count}"):
        assert key not in out


def test_quality_prompt_shows_the_model_real_cloze_syntax_and_a_json_example():
    out = quality.QUALITY_SYSTEM.format(language_instruction="Write in English.")
    assert "{{c1::Dura mater}}" in out                 # double braces survive formatting
    assert '"cards": [' in out and "{" in out          # the JSON example is intact


# ── the quality step itself, end to end, with a fake Gemini ─────────────────────

class _FakeQualityClient:
    def __init__(self):
        self.systems = []
        self.models = self

    def generate_content(self, model, contents, config=None):
        self.systems.append(config["system_instruction"])
        import json
        cards = [{"front": "Better question?", "back": "• a\n• b\n• c\n💡 ABC", "type": "basic", "tags": ["x"],
                  "improved": True, "mnemonic_added": True}]
        return types.SimpleNamespace(text=json.dumps({"cards": cards}))


def test_quality_step_runs_instead_of_crashing(monkeypatch):
    fake = _FakeQualityClient()
    monkeypatch.setattr(quality.genai, "Client", lambda api_key: fake)
    monkeypatch.setattr(quality.time, "sleep", lambda s: None)
    cards = [{"front": "q", "back": "a", "type": "basic", "tags": ["x"]}]
    improved, res = quality.improve_cards(cards, "D", api_key="k", use_pro=False, lang="en")
    assert res.mnemonics_added == 1 and improved[0]["front"] == "Better question?"
    assert "improved" not in improved[0] and "mnemonic_added" not in improved[0]
    assert "Write every" in fake.systems[0] and "{{c1::Dura mater}}" in fake.systems[0]
