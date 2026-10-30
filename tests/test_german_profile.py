"""
Profiles: "medical" (default, unchanged) and "german" (textbook vocabulary & grammar).

Before this, the web pipeline had ONE set of medical prompts for everything. The German
prompt lived only in pdf_gen.py, reachable from the old terminal menu, never from the
web UI. Worse: with Card language = Uzbek, a safety net would have transliterated the
German words themselves into Cyrillic.
"""
import json
import os
import types

import pytest

import anki_gen
import chunker
import filter as flt
import format_pass
import generator
import pipeline
import profiles
import quality
import usage
from chunker import Chunk
from fakes import (quality_stub, FakeChatClient, FakeGemini, Sleeper, Resp, gemini_client, groq_client)

KW = dict(chunk_index=1, chunk_total=3, topic="Berufe", language_instruction="LANG-INSTRUCTION", max_cards=40)

GERMAN_CARDS = [
    {"front": "der Arzt / die Ärztin", "back": "doctor<br>Plural: die Ärzte<br>Der Arzt untersucht den Patienten.",
     "type": "basic", "tags": ["vocabulary", "nouns"]},
    {"front": "Ich {{c1::bin}} ins Kino gegangen.", "back": "Perfekt von gehen: ist gegangen", "type": "cloze", "tags": ["grammar"]},
    {"front": "die Krankenschwester", "back": "nurse<br>Plural: die Krankenschwestern", "type": "basic", "tags": ["vocabulary"]},
]


def pdf_chunk(text="Lektion 3 Berufe der Arzt die Ärzte"):
    return Chunk(text=text, index=1, total=1, source="pdf", page_from=1, page_to=2)


# ── profile registry ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [(None, "medical"), ("", "medical"), ("medical", "medical"),
                                             ("german", "german"), ("  German ", "german")])
def test_normalize_profile(value, expected):
    assert profiles.normalize_profile(value) == expected


@pytest.mark.parametrize("value", ["french", "lang", 5])
def test_unknown_profile_is_rejected(value):
    with pytest.raises(ValueError):
        profiles.normalize_profile(value)


# ── what "Card language" means for a language deck ──────────────────────────────

@pytest.mark.parametrize("lang, must_contain", [("en", "English"), ("ru", "Russian"), ("uz", "CYRILLIC"), ("de", "simple German")])
def test_explanation_instruction_keeps_the_german_and_translates_only_the_explanation(lang, must_contain):
    text = profiles.build_explanation_instruction(lang)
    assert must_contain in text
    assert "stay in German" in text or lang == "de"
    assert "Write every" not in text                  # the medical instruction would translate the German front too


def test_the_medical_instruction_would_have_translated_the_german_front():
    # documents WHY the german profile needs its own instruction
    assert "Write every" in anki_gen.build_language_instruction("uz")


# ── the prompts themselves ──────────────────────────────────────────────────────

@pytest.mark.parametrize("tpl", [profiles.GERMAN_SYSTEM_PROMPT, profiles.GERMAN_FALLBACK_PROMPT, profiles.GERMAN_QUALITY_SYSTEM])
def test_german_templates_format_cleanly(tpl):
    out = tpl.format(**KW)
    assert "{language_instruction}" not in out and "{topic}" not in out and "{chunk_index}" not in out
    assert "{{c1::" in out                             # real Anki cloze syntax reaches the model
    assert "___" not in out.replace("(___)", "").replace('"___"', "")      # only ever mentioned as a thing NOT to write


@pytest.mark.parametrize("tpl", [profiles.GERMAN_SYSTEM_PROMPT, profiles.GERMAN_FALLBACK_PROMPT])
def test_generation_prompts_cover_what_a_textbook_deck_needs(tpl):
    out = tpl.format(**KW)
    low = out.lower()
    for must in ("article", "plural", "perfekt"):
        assert must in low
    assert "never guess" in low                        # a wrong plural that gets memorised is the worst outcome
    assert "never pad" in low and "minimum" not in low
    assert "Ergänzen Sie" in out                       # exercise instructions are to be ignored
    assert "markdown" in low and "<b>" in out
    assert "medical" not in low
    assert "LANG-INSTRUCTION" in out


def test_quality_prompt_checks_grammar_and_protects_the_german_front():
    out = profiles.GERMAN_QUALITY_SYSTEM.format(**KW).lower()
    for must in ("article", "plural", "partizip", "do not remove cards", "never translate the front"):
        assert must in out
    assert "medical" not in out and "mnemonic" in out and "do not add mnemonics" in out


def test_dedup_prompt_keeps_the_schema_the_parser_expects_and_protects_cloze_of_a_vocab_card():
    out = profiles.GERMAN_DEDUP_SYSTEM
    assert '"results"' in out and '"keep"' in out and '"remove"' in out
    assert "NOT duplicates" in out and "cloze" in out.lower()


# ── generation: the profile reaches the model ───────────────────────────────────

def test_gemini_gets_the_german_prompt_and_an_instruction_that_does_not_translate_the_front():
    fake = FakeGemini()
    anki_gen.generate_cards("text", "Deutsch", fake, 1, 2, "Berufe", "uz", profile="german")
    system = fake.systems[0]
    assert "German-as-a-foreign-language" in system and "CYRILLIC" in system and "Write every" not in system


def test_default_profile_is_still_medical():
    fake = FakeGemini()
    anki_gen.generate_cards("text", "Neuro", fake, 1, 2, "CSF", "en")
    assert "medical Anki card creator" in fake.systems[0]


@pytest.mark.parametrize("fn", [generator._generate_with_groq, generator._generate_with_openrouter])
def test_fallback_providers_get_the_german_prompt_too(fn):
    fb = FakeChatClient()
    fn(pdf_chunk(), "Deutsch", groq_client(fb), "ru", "german")
    assert "German-as-a-foreign-language" in fb.systems[0] and "Russian" in fb.systems[0]
    fb2 = FakeChatClient()
    fn(pdf_chunk(), "Neuro", groq_client(fb2), "en")
    assert "medical" in fb2.systems[0].lower()


def test_topic_detection_asks_for_a_vocabulary_theme_or_grammar_point():
    fake = FakeGemini()
    generator.detect_topic(pdf_chunk(), gemini_client(fake), profile="german")
    assert "German textbook" in fake.topic_prompts[0] and "grammar point" in fake.topic_prompts[0]
    fake = FakeGemini()
    generator.detect_topic(pdf_chunk(), gemini_client(fake))
    assert fake.topic_prompts and "anatomical or medical topic" in fake.topic_prompts[0]


def test_generate_chunk_and_run_pipeline_pass_the_profile_down(monkeypatch):
    monkeypatch.setattr(generator, "time", types.SimpleNamespace(sleep=Sleeper()))
    fake = FakeGemini()
    res = generator._run_pipeline([pdf_chunk()], "Deutsch", [gemini_client(fake)], True, None, "ru", "german")
    assert res.chunks_done == 1
    assert "German-as-a-foreign-language" in fake.systems[0] and "German textbook" in fake.topic_prompts[0]


def test_fallback_is_used_with_the_german_prompt_when_gemini_fails(monkeypatch):
    monkeypatch.setattr(generator, "time", types.SimpleNamespace(sleep=Sleeper()))
    gem = FakeGemini(lambda idx, n: Exception("500 down"))
    fb = FakeChatClient()
    cards, who = generator.generate_chunk(pdf_chunk(), "Deutsch", [gemini_client(gem), groq_client(fb)], "en", "german")
    assert who == "Groq Llama" and "German-as-a-foreign-language" in fb.systems[0]


@pytest.mark.parametrize("lang, profile, expected", [
    ("en", "medical", ["en"]), ("ru", "medical", ["ru", "en"]),
    ("en", "german", ["de", "en"]), ("ru", "german", ["de", "ru", "en"]), ("de", "german", ["de", "en"]),
])
def test_german_lessons_prefer_the_german_caption_track(lang, profile, expected):
    assert generator._transcript_language_priority(lang, profile) == expected


def test_generate_from_pdf_forwards_the_profile(monkeypatch):
    seen = {}
    monkeypatch.setattr(generator, "chunk_pdf", lambda *a, **k: [pdf_chunk()])
    monkeypatch.setattr(generator, "_run_pipeline", lambda *a, **k: seen.update(args=a) or generator.GenerationResult([], 0, 0))
    generator.generate_from_pdf("x.pdf", "D", [], profile="german")
    assert seen["args"][-1] == "german"


# ── filter: no medical "trivial" verdicts on vocabulary ─────────────────────────

class _GroqSpy:
    """Groq stand-in: records every system prompt; keeps all cards; removes nothing."""
    def __init__(self, api_key=None):
        self.systems = []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model, messages, temperature):
        self.systems.append(messages[0]["content"])
        user = messages[1]["content"]
        if messages[0]["content"] == flt.FILTER_SYSTEM:
            cards = json.loads(user[user.index("Cards:\n") + 7: user.rindex("\n\nReturn")])
            out = {"kept": cards, "removed": []}
        else:
            out = {"results": []}
        msg = types.SimpleNamespace(content=json.dumps(out))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_german_vocabulary_is_not_sent_to_the_medical_quality_filter(monkeypatch):
    spy = _GroqSpy()
    monkeypatch.setattr(flt, "Groq", lambda api_key: spy)
    monkeypatch.setattr(flt.time, "sleep", lambda s: None)
    res = flt.filter_cards(GERMAN_CARDS, "Deutsch", "key", profile="german")
    assert flt.FILTER_SYSTEM not in spy.systems                       # "trivial for a medical student" never asked
    assert res.cards_in == 3 and res.cards_out == 3
    # dedup (if any request is made) uses the German rules
    assert all(s == profiles.GERMAN_DEDUP_SYSTEM for s in spy.systems)


def test_medical_profile_still_uses_the_medical_filter(monkeypatch):
    spy = _GroqSpy()
    monkeypatch.setattr(flt, "Groq", lambda api_key: spy)
    monkeypatch.setattr(flt.time, "sleep", lambda s: None)
    flt.filter_cards(GERMAN_CARDS, "Neuro", "key")
    assert flt.FILTER_SYSTEM in spy.systems


def test_german_dedup_uses_the_german_rules_for_similar_cards(monkeypatch):
    spy = _GroqSpy()
    monkeypatch.setattr(flt, "Groq", lambda api_key: spy)
    monkeypatch.setattr(flt.time, "sleep", lambda s: None)
    cards = [{"front": "die Krankenschwester", "back": "nurse<br>Plural: die Krankenschwestern", "type": "basic", "tags": []},
             {"front": "Die {{c1::Krankenschwester}} arbeitet im Krankenhaus.", "back": "nurse", "type": "cloze", "tags": []}]
    flt.filter_cards(cards, "Deutsch", "key", profile="german")
    assert spy.systems == [profiles.GERMAN_DEDUP_SYSTEM]              # asked, with the German rules


# ── quality ─────────────────────────────────────────────────────────────────────

class _FakeQuality:
    def __init__(self):
        self.systems, self.models = [], self

    def generate_content(self, model, contents, config=None):
        self.systems.append(config["system_instruction"])
        return types.SimpleNamespace(text=json.dumps({"cards": json.loads(contents[contents.index("Cards:\n") + 7: contents.rindex("\n\nReturn")])}))


@pytest.mark.parametrize("profile, marker", [("german", "teacher of German"), ("medical", "senior medical educator")])
def test_quality_prompt_follows_the_profile(monkeypatch, profile, marker):
    fake = _FakeQuality()
    monkeypatch.setattr(quality.genai, "Client", lambda api_key: fake)
    monkeypatch.setattr(quality.time, "sleep", lambda s: None)
    quality.improve_cards([dict(c) for c in GERMAN_CARDS], "D", api_key="k", use_pro=False, lang="uz", profile=profile)
    assert marker in fake.systems[0]
    assert ("Write every" in fake.systems[0]) == (profile == "medical")


# ── format pass: the Uzbek-Cyrillic safety net must not touch German ────────────

def test_uzbek_safety_net_does_not_transliterate_german_words():
    cards, _ = format_pass.format_cards([dict(c) for c in GERMAN_CARDS], "D", "", "uz", "german")
    assert cards[0]["front"] == "der Arzt / die Ärztin"
    assert "Plural: die Ärzte" in cards[0]["back"]
    assert "Ich {{c1::bin}} ins Kino gegangen." == cards[1]["front"]


def test_medical_uzbek_decks_are_still_transliterated():
    card = {"front": "Choroid plexus", "back": "Phrenic nerve", "type": "basic", "tags": []}
    cards, res = format_pass.format_cards([card], "D", "", "uz", "medical")
    assert format_pass.has_cyrillic(cards[0]["front"]) and format_pass.has_cyrillic(cards[0]["back"])
    assert res.uz_transliterated == 2


# ── whole pipeline (PDF source) ─────────────────────────────────────────────────

@pytest.fixture
def env(tmp_path, monkeypatch):
    s = Sleeper()
    real_time = __import__("time").time
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(usage, "USAGE_FILE", str(tmp_path / "usage.json"))
    for mod in (pipeline, generator):
        monkeypatch.setattr(mod, "time", types.SimpleNamespace(sleep=s, time=real_time))
    monkeypatch.setattr(generator, "chunk_pdf", lambda *a, **k: [pdf_chunk()])
    cfg = {"gemini_api_key": "k"}
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)

    def configure(raw_reply, extra_cfg=None):
        cfg.update(extra_cfg or {})
        gem = FakeGemini(lambda idx, n: raw_reply)
        monkeypatch.setattr(pipeline, "build_clients", lambda c: [gemini_client(gem)])
        return gem

    configure.cfg = cfg
    return configure


def _saved_rows(res):
    import csv
    with open(res.filename, encoding="utf-8", newline="") as f:
        return list(csv.reader([ln for ln in f.read().splitlines(True) if not ln.startswith("#")], delimiter="\t"))


def _run(**kw):
    return pipeline.run(pipeline.PipelineConfig(source="pdf", input="Menschen_A1.pdf", deck="Deutsch", run_filter=False,
                                                run_quality=False, **kw))


# plain card first, then one the medical heuristic would rank first (bullets, 💡, digits)
ORDERED = [{"front": "der Tisch", "back": "table", "type": "basic", "tags": []},
           {"front": "die Zahl", "back": "• number<br>• 1, 2, 3 💡 remember: eins zwei drei vier", "type": "basic", "tags": []}]


def test_german_deck_keeps_textbook_order_and_german_fronts_with_uzbek_explanations(env):
    env(json.dumps({"cards": ORDERED}))
    res = _run(profile="german", lang="uz")
    assert [r[1] for r in _saved_rows(res)] == ["der Tisch", "die Zahl"]       # not re-sorted by the medical score
    assert res.stages["profile"] == "german"


def test_medical_decks_are_still_sorted_by_score(env):
    env(json.dumps({"cards": ORDERED}))
    res = _run(profile="medical", lang="en")
    assert [r[1] for r in _saved_rows(res)] == ["die Zahl", "der Tisch"]


def test_profile_reaches_every_step(env, monkeypatch):
    env(json.dumps({"cards": ORDERED}), {"groq_api_key": "g", "gemini_pro_key": "p"})
    seen = {}
    monkeypatch.setattr(pipeline, "filter_cards", lambda **kw: seen.update(filter=kw["profile"]) or
                        types.SimpleNamespace(cards_in=2, cards_out=2, reasons={}, _kept_cards=kw["cards"]))
    monkeypatch.setattr(pipeline, "get_kept_cards", lambda r: r._kept_cards)
    monkeypatch.setattr(pipeline, "improve_cards", lambda **kw: seen.update(quality=kw["profile"]) or quality_stub(kw["cards"]))
    real_format = pipeline.format_cards
    monkeypatch.setattr(pipeline, "format_cards", lambda **kw: seen.update(format=kw["profile"]) or real_format(**kw))
    pipeline.run(pipeline.PipelineConfig(source="pdf", input="x.pdf", deck="D", profile="german", lang="ru"))
    assert seen == {"filter": "german", "quality": "german", "format": "german"}


def test_unknown_profile_fails_early_with_a_clear_message(env):
    env("{}")
    with pytest.raises(ValueError, match="profile"):
        _run(profile="klingon")


def test_run_wrappers_accept_a_profile(env, monkeypatch):
    captured = {}
    monkeypatch.setattr(pipeline, "run", lambda config, on_progress=None: captured.update(p=config.profile))
    pipeline.run_pdf("x.pdf", "D", profile="german")
    assert captured["p"] == "german"
    pipeline.run_youtube("https://youtu.be/dQw4w9WgXcQ", "D", profile="german")
    assert captured["p"] == "german"
    pipeline.run_video("x.mp4", "D", profile="german")
    assert captured["p"] == "german"


# ── server + UI ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("route, body", [
    ("/api/generate/youtube", {"youtube_url": "https://youtu.be/dQw4w9WgXcQ"}),
    ("/api/generate/pdf", {"pdf_path": __file__}),
])
def test_server_forwards_the_profile_and_rejects_unknown_ones(monkeypatch, route, body):
    import server
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return types.SimpleNamespace(total_final=0, total_raw=0, total_filtered=0, filename="x.txt", duration_sec=0.1,
                                     stages={}, cards=[], errors=[])

    monkeypatch.setattr(server, "run_youtube", fake_run)
    monkeypatch.setattr(server, "run_pdf", fake_run)
    client = server.app.test_client()

    assert client.post(route, json=body).status_code == 200 and seen["profile"] == "medical"
    assert client.post(route, json={**body, "profile": "german"}).status_code == 200 and seen["profile"] == "german"
    r = client.post(route, json={**body, "profile": "klingon"})
    assert r.status_code == 400 and "profile" in r.get_json()["error"].lower()


def test_ui_has_a_content_type_picker_and_sends_it():
    html = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html"), encoding="utf-8").read()
    assert 'id="profileGroup"' in html
    assert "selectChip('profile', this, 'medical')" in html and "selectChip('profile', this, 'german')" in html
    assert "let currentProfile" in html and "profile:     currentProfile" in html


# ── end to end into a REAL Anki collection ──────────────────────────────────────

def test_german_deck_imports_into_real_anki_with_umlauts_and_working_cloze(env, tmp_path):
    pytest.importorskip("anki")
    from anki.collection import Collection
    from anki.import_export_pb2 import ImportCsvRequest

    reply = json.dumps({"cards": [
        {"front": "der Arzt / die Ärztin", "back": "шифокор<br>Plural: die Ärzte / die Ärztinnen<br>Der Arzt untersucht den Patienten.",
         "type": "basic", "tags": ["vocabulary", "nouns"]},
        {"front": "Ich {{c1::bin}} gestern ins Kino gegangen.", "back": "Perfekt: ist gegangen", "type": "cloze", "tags": ["grammar", "perfekt"]},
        # a model that ignored the rule and wrote the old-style blank:
        {"front": "Er ___ Lehrer geworden. (ist)", "back": "werden: ist geworden", "type": "cloze", "tags": ["grammar"]},
    ]})
    env(reply)
    res = _run(profile="german", lang="uz")
    assert not res.errors

    path = os.path.abspath(res.filename)
    col = Collection(str(tmp_path / "c.anki2"))
    try:
        log = col.import_csv(ImportCsvRequest(path=path, metadata=col.get_csv_metadata(path, None)))
        assert len(log.log.new) == 3
        notes = [col.get_note(n) for n in sorted(col.find_notes(""))]
        kinds = [(n.note_type()["name"], len(n.card_ids())) for n in notes]
        assert kinds == [("Basic", 1), ("Cloze", 1), ("Cloze", 1)]          # every cloze really is a cloze
        assert notes[0].fields[0] == "der Arzt / die Ärztin"                 # German untouched, umlauts intact
        assert "Plural: die Ärzte / die Ärztinnen" in notes[0].fields[1]
        assert "шифокор" in notes[0].fields[1]                               # the translation stayed Cyrillic Uzbek
        assert "{{c1::" in notes[2].fields[0] and "___" not in notes[2].fields[0]   # legacy blank repaired
    finally:
        col.close()



# ── how many cards a chunk may produce ──────────────────────────────────────────

@pytest.mark.parametrize("chars, cap", [(0, 40), (1000, 40), (1400, 46), (3299, 109), (4200, 120), (5761, 120), (100_000, 120)])
def test_card_cap_grows_with_the_size_of_the_chunk(chars, cap):
    assert profiles.german_card_cap("x" * chars) == cap


def test_a_dense_vocabulary_chunk_is_not_capped_at_40_cards():
    """A Wortschatz page: ~100 entries in 3.3k characters. A fixed cap of 40 covered 40% of them."""
    text = "\n".join(f"der Wort{i}, die Worte – word {i}" for i in range(100))
    fake = FakeGemini()
    anki_gen.generate_cards(text, "Deutsch", fake, 1, 4, "Wortschatz", "ru", profile="german")
    assert f"up to {profiles.german_card_cap(text)} per chunk" in fake.systems[0]
    assert profiles.german_card_cap(text) > 40 and "up to 40" not in fake.systems[0]


@pytest.mark.parametrize("fn", [generator._generate_with_groq, generator._generate_with_openrouter])
def test_fallback_providers_get_the_same_cap(fn):
    text = "x" * 3299
    fb = FakeChatClient()
    fn(pdf_chunk(text), "Deutsch", groq_client(fb), "ru", "german")
    assert f"up to {profiles.german_card_cap(text)}." in fb.systems[0] or f"up to {profiles.german_card_cap(text)}," in fb.systems[0]


def test_medical_prompts_are_unchanged():
    fake = FakeGemini()
    anki_gen.generate_cards("x" * 5000, "Neuro", fake, 1, 2, "CSF", "en")
    assert "up to 35 per chunk" in fake.systems[0]
