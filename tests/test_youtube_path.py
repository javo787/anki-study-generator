"""
YouTube -> cards path, tested offline (no network, no API keys).
Every test here corresponds to a defect reproduced in the audit.
"""
import json
import random
import types

import pytest

import anki_gen
import chunker
import generator
import pipeline
import usage
from fakes import (DAILY_429, RATE_429, FakeChatClient, FakeGemini, Sleeper, block_entries,
                   gemini_client, groq_client)

ID = "dQw4w9WgXcQ"


# ── 1. video URL parsing ────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={ID}",
    f"https://www.youtube.com/watch?feature=share&v={ID}&t=95s",
    f"https://www.youtube.com/watch?v={ID}&list=PLabcdefghijklmnopqrstuvwxyz012345&index=3",
    f"https://m.youtube.com/watch?v={ID}",
    f"https://music.youtube.com/watch?v={ID}",
    f"https://youtu.be/{ID}",
    f"https://youtu.be/{ID}?si=AbCdEfGhIjKlMnOp&t=30",
    f"https://www.youtube.com/embed/{ID}",
    f"https://www.youtube-nocookie.com/embed/{ID}",
    f"https://www.youtube.com/shorts/{ID}",
    f"https://www.youtube.com/live/{ID}?feature=share",
    f"https://www.youtube.com/v/{ID}",
    f"youtube.com/watch?v={ID}",
    f"  https://youtu.be/{ID}\n",
    ID,
    f"  {ID}\n",
])
def test_video_id_is_extracted(url):
    assert anki_gen.extract_video_id(url) == ID


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/playlist?list=PLabcdefghijklmnopqrstuvwxyz012345",
    "https://www.youtube.com/@DrNajeeb",
    "https://example.com/page?dev=abcdefghijk",                       # not YouTube at all
    "https://example.com/watch?v=dQw4w9WgXcQ",                        # right shape, wrong site
    "",
    "not a url",
])
def test_non_video_urls_are_rejected(url):
    with pytest.raises(ValueError):
        anki_gen.extract_video_id(url)


def test_v_must_be_the_v_parameter_not_a_suffix_of_another():
    assert anki_gen.extract_video_id(f"https://www.youtube.com/watch?abv=ZZZZZZZZZZZ&v={ID}") == ID


# ── 2. transcript cleaning ──────────────────────────────────────────────────────

class _Snip:
    def __init__(self, text, start, duration=2.0):
        self.text, self.start, self.duration = text, start, duration


class _FakeApi:
    snippets = []

    def __init__(self, proxy_config=None):
        pass

    def fetch(self, video_id, languages=("en",)):
        return _Fetched(self.snippets)


class _Fetched(list):
    language_code = "en"


def test_sound_tags_and_speaker_markers_are_removed(monkeypatch):
    _FakeApi.snippets = [
        _Snip("[Music]", 0.0), _Snip(">> so the choroid plexus", 2.0), _Snip("[Applause]", 4.0),
        _Snip("produces CSF [Music]", 6.0), _Snip("  ", 8.0), _Snip("♪♪", 10.0),
        _Snip("the line\nbreaks here", 12.0),
    ]
    monkeypatch.setattr(anki_gen, "YouTubeTranscriptApi", _FakeApi)
    got = anki_gen.get_transcript_entries(ID)
    assert [e["text"] for e in got] == ["so the choroid plexus", "produces CSF", "the line breaks here"]
    assert [e["start"] for e in got] == [2.0, 6.0, 12.0]


# ── 3. chunking: no tiny tail chunk ─────────────────────────────────────────────

def test_short_outro_is_merged_into_previous_chunk_not_sent_alone():
    body = block_entries(n_blocks=1, block_seconds=590)               # ~9.8 min of speech
    end = body[-1]["start"] + body[-1]["duration"]
    outro = [{"text": "thank you for watching see you next time", "start": end + 4.0, "duration": 4.0}]
    chunks = chunker.chunk_youtube(body + outro)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("see you next time")
    assert chunks[0].index == 1 and chunks[0].total == 1


@pytest.mark.parametrize("seed", range(40))
def test_chunks_are_lossless_ordered_and_never_tiny(seed):
    rng = random.Random(seed)
    entries, t = [], 0.0
    for k in range(rng.randint(60, 700)):
        dur = rng.uniform(2, 6)
        entries.append({"text": f"w{k} filler words of the lecture", "start": round(t, 2), "duration": dur + rng.choice([0, 2])})
        t += dur + (rng.uniform(3, 9) if rng.random() < 0.08 else 0)
    chunks = chunker.chunk_youtube(entries)

    assert " ".join(c.text for c in chunks).split() == " ".join(e["text"] for e in entries).split()
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.total == len(chunks) for c in chunks)
    assert all(a.end_time <= b.start_time + 10 for a, b in zip(chunks, chunks[1:]))
    if len(chunks) > 1:
        assert all((c.end_time - c.start_time) >= chunker.MIN_CHUNK_SECONDS - 6 for c in chunks)
    assert all((c.end_time - c.start_time) <= chunker.MAX_CHUNK_SECONDS + chunker.MIN_CHUNK_SECONDS + 10 for c in chunks)


# ── 4. rate limits: wait as the API asks, then retry ────────────────────────────

@pytest.fixture
def sleeper(monkeypatch):
    s = Sleeper()
    monkeypatch.setattr(generator.time, "sleep", s)
    return s


def _one_chunk():
    return chunker.chunk_youtube(block_entries(n_blocks=1))[0]


def test_429_waits_for_the_requested_delay_then_succeeds_on_the_same_provider(sleeper):
    fake = FakeGemini(lambda idx, n: Exception(RATE_429) if n <= 2 else None)
    cards, who = generator.generate_chunk(_one_chunk(), "D", [gemini_client(fake)], "en")
    assert len(cards) == 3 and who.startswith("Gemini")
    assert fake.attempts == {1: 3}
    assert sleeper.calls == [pytest.approx(42.2, abs=1.5)] * 2          # 41.2 s asked + 1 s margin


def test_daily_quota_429_does_not_wait_and_falls_back_immediately(sleeper):
    fake, fb = FakeGemini(lambda idx, n: Exception(DAILY_429)), FakeChatClient()
    cards, who = generator.generate_chunk(_one_chunk(), "D", [gemini_client(fake), groq_client(fb)], "en")
    assert who == "Groq Llama" and sleeper.calls == [] and fake.attempts == {1: 1}


def test_429_with_an_unreasonably_long_delay_falls_back_without_waiting(sleeper):
    long_wait = RATE_429.replace("41.2s", "900s").replace("'41s'", "'900s'")
    fake, fb = FakeGemini(lambda idx, n: Exception(long_wait)), FakeChatClient()
    _, who = generator.generate_chunk(_one_chunk(), "D", [gemini_client(fake), groq_client(fb)], "en")
    assert who == "Groq Llama" and sleeper.calls == []


def test_persistent_429_gives_up_after_three_attempts_and_falls_back(sleeper):
    fake, fb = FakeGemini(lambda idx, n: Exception(RATE_429)), FakeChatClient()
    _, who = generator.generate_chunk(_one_chunk(), "D", [gemini_client(fake), groq_client(fb)], "en")
    assert fake.attempts == {1: 3} and who == "Groq Llama" and len(sleeper.calls) == 2


def test_503_behaviour_is_unchanged(sleeper):
    fake = FakeGemini(lambda idx, n: Exception("503 UNAVAILABLE overloaded") if n < 3 else None)
    cards, _ = generator.generate_chunk(_one_chunk(), "D", [gemini_client(fake)], "en")
    assert len(cards) == 3 and sleeper.calls == [15, 30]


# ── 5. a failed chunk gets a second chance; leftovers are reported ──────────────

def _chunks(n=5):
    return chunker.chunk_youtube(block_entries(n_blocks=n))


def test_failed_chunk_is_retried_in_a_second_pass(sleeper):
    # chunk 2 fails on its first call (non-429 error, no fallback configured), works afterwards
    fake = FakeGemini(lambda idx, n: Exception("500 internal") if (idx == 2 and n == 1) else None)
    res = generator._run_pipeline(_chunks(), "D", [gemini_client(fake)], True, None, "en")
    assert res.chunks_total == 5 and res.chunks_done == 5 and res.errors == []
    assert generator.SECOND_PASS_COOLDOWN in sleeper.calls


def test_chunks_that_still_fail_are_reported_with_their_time_range(sleeper):
    fake = FakeGemini(lambda idx, n: Exception("500 internal") if idx in (2, 4) else None)
    res = generator._run_pipeline(_chunks(), "D", [gemini_client(fake)], True, None, "en")
    assert res.chunks_done == 3 and len(res.errors) == 2
    assert "Chunk 2" in res.errors[0] and "05:03" in res.errors[0]       # time range of the lost part
    assert res.failed_chunks == [2, 4]


def test_no_second_pass_when_everything_worked(sleeper):
    res = generator._run_pipeline(_chunks(), "D", [gemini_client(FakeGemini())], True, None, "en")
    assert res.chunks_done == 5 and sleeper.calls == []


# ── 6. pipeline.run: honest result, correct usage keys, no pointless 60 s wait ──

@pytest.fixture
def env(tmp_path, monkeypatch):
    s = Sleeper()
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(usage, "USAGE_FILE", str(tmp_path / "usage.json"))
    monkeypatch.setattr(pipeline, "time", types.SimpleNamespace(sleep=s, time=__import__("time").time))
    monkeypatch.setattr(generator, "time", types.SimpleNamespace(sleep=s, time=__import__("time").time))
    monkeypatch.setattr(generator, "get_transcript_entries", lambda *a, **k: block_entries(n_blocks=4))
    monkeypatch.setattr(pipeline, "build_proxy_config", lambda cfg: None)
    monkeypatch.setattr(pipeline, "improve_cards",
                        lambda **kw: (kw["cards"], types.SimpleNamespace(mnemonics_added=0, phrasing_improved=0, model_used="fake")))

    def configure(cfg, fake):
        monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
        monkeypatch.setattr(pipeline, "build_clients", lambda c: [gemini_client(fake)])

    configure.sleeps = s
    configure.tmp = tmp_path
    return configure


def _run(**kw):
    cfg = pipeline.PipelineConfig(source="youtube", input=f"https://youtu.be/{ID}", deck="CSF",
                                  run_filter=False, run_quality=kw.pop("run_quality", False), **kw)
    return pipeline.run(cfg)


def test_partial_coverage_is_reported_not_silent(env):
    env({"gemini_api_key": "k"}, FakeGemini(lambda idx, n: Exception("500 boom") if idx in (2, 3) else None))
    res = _run()
    gen = res.stages["generation"]
    assert gen["chunks_total"] == 4 and gen["chunks"] == 2 and gen["chunks_failed"] == 2
    assert gen["coverage_pct"] == 50
    assert any("2 of 4" in e for e in res.errors), res.errors             # headline warning
    assert sum("Chunk" in e for e in res.errors) == 2                     # which parts were lost


def test_full_success_has_no_errors(env):
    env({"gemini_api_key": "k"}, FakeGemini())
    res = _run()
    assert res.errors == [] and res.stages["generation"]["coverage_pct"] == 100


def test_usage_is_recorded_under_the_keys_the_monitor_shows_and_never_as_none(env):
    env({"gemini_api_key": "k"}, FakeGemini(lambda idx, n: Exception("500 boom") if idx == 1 else None))
    _run()
    providers = json.load(open(usage.USAGE_FILE))["providers"]
    assert set(providers) == {"gemini_flash"}
    assert providers["gemini_flash"]["requests"] == 3                     # only the 3 chunks that succeeded
    assert usage.get_all_usage()["gemini_flash"]["requests_used"] == 3


@pytest.mark.parametrize("cfg, quality", [
    ({"gemini_api_key": "k"}, False),
    ({"gemini_api_key": "k", "gemini_pro_key": "p"}, True),      # quality runs on gemini-2.5-pro: its own quota
])
def test_no_fixed_60s_wait_after_generation(env, cfg, quality):
    env(cfg, FakeGemini())
    _run(run_quality=quality)
    assert 60 not in env.sleeps.calls


# ── 7. prompts ──────────────────────────────────────────────────────────────────

KW = dict(chunk_index=1, chunk_total=3, topic="CSF", language_instruction="English")


def test_prompts_format_cleanly():
    for tpl in (anki_gen.SYSTEM_PROMPT, generator.GROQ_CARD_PROMPT):
        out = tpl.format(**KW)
        assert "{topic}" not in out and "{chunk_index}" not in out and "{language_instruction}" not in out
        assert "{{c1::" in out                                        # braces survived formatting intact


def test_every_provider_prompt_warns_about_speech_recognition_errors_and_does_not_pad():
    for tpl in (anki_gen.SYSTEM_PROMPT, generator.GROQ_CARD_PROMPT):
        out = tpl.format(**KW)
        assert "speech-recognition" in out and "choroid" in out.lower()
        assert "Minimum 15" not in out and "minimum 15" not in out.lower()
        assert "never pad" in out.lower() or "do not pad" in out.lower()


def test_fallback_prompt_teaches_real_cloze_syntax_and_html():
    out = generator.GROQ_CARD_PROMPT.format(**KW)
    assert "{{c1::" in out and '"___"' not in out and "___ is" not in out
    assert "<b>" in out and "markdown" in out.lower()
