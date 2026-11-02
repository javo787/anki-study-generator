"""
Resume: a run that failed half way must not re-spend the quota of the chunks that worked.

Offline; the same fakes as the rest of the suite.
"""
import copy
import json
import os
import time
import types

import pytest

import anki_gen
import chunk_cache
import chunker
import generator
import pipeline
import usage
import video_gen
from fakes import FakeGemini, Sleeper, block_entries, gemini_client

ID = "dQw4w9WgXcQ"


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    s = Sleeper()
    monkeypatch.setattr(generator, "time", types.SimpleNamespace(sleep=s, time=time.time))
    return s


def make_chunks(n=4, video_id=ID):
    chunks = chunker.chunk_youtube(block_entries(n_blocks=n))
    for c in chunks:
        c.video_id = video_id
    return chunks


def cache(tmp_path, deck="D", lang="en", profile="medical", detect_topics=True):
    return generator.make_chunk_cache(str(tmp_path / "cache"), deck, lang, profile, detect_topics)


def run(chunks, fake, cch, deck="D", lang="en", profile="medical", detect=True):
    return generator._run_pipeline(chunks, deck, [gemini_client(fake)], detect, None, lang, profile, cache=cch)


def calls(fake):
    return sum(fake.attempts.values())


# ── reuse ───────────────────────────────────────────────────────────────────────

def test_a_second_run_reuses_every_chunk_and_calls_no_model(tmp_path):
    cch = cache(tmp_path)
    first_fake = FakeGemini()
    first = run(make_chunks(), first_fake, cch)
    assert first.cache_hits == 0 and calls(first_fake) == 4 and len(os.listdir(tmp_path / "cache" / "chunks")) == 4

    fake = FakeGemini()
    second = run(make_chunks(), fake, cache(tmp_path))
    assert calls(fake) == 0 and fake.topic_prompts == []                       # not even the topic-detection call
    assert second.cache_hits == 4 and second.chunks_done == 4 and second.ai_used == {"cache": 4}
    assert [c["front"] for c in second.cards] == [c["front"] for c in first.cards]
    assert [c["t"] for c in second.cards] == [c["t"] for c in first.cards]     # source times are recomputed, not stale


def test_resume_after_a_failure_only_asks_for_the_missing_chunk(tmp_path):
    broken = FakeGemini(lambda idx, n: Exception("500 internal") if idx == 2 else None)
    first = run(make_chunks(), broken, cache(tmp_path))
    assert first.chunks_done == 3 and first.failed_chunks == [2]               # (second pass failed too)

    healthy = FakeGemini()
    second = run(make_chunks(), healthy, cache(tmp_path))
    assert healthy.attempts == {2: 1}                                         # exactly the chunk that was missing
    assert second.chunks_done == 4 and second.failed_chunks == [] and second.cache_hits == 3
    assert second.errors == []


def test_chunks_that_failed_are_not_cached(tmp_path):
    run(make_chunks(), FakeGemini(lambda idx, n: Exception("500") if idx == 3 else None), cache(tmp_path))
    assert len(os.listdir(tmp_path / "cache" / "chunks")) == 3


def test_cached_cards_are_handed_out_as_copies_and_tags_do_not_pile_up(tmp_path):
    run(make_chunks(), FakeGemini(), cache(tmp_path))
    seen = []
    for _ in range(3):
        res = run(make_chunks(), FakeGemini(), cache(tmp_path))
        seen.append([c["tags"] for c in res.cards])
        res.cards[0]["tags"].append("mutated-by-the-caller")                    # must not leak into the cache
    assert seen[0] == seen[1] == seen[2]
    assert all(len(set(tags)) == len(tags) for tags in seen[0])


def test_the_topic_is_cached_with_the_cards(tmp_path):
    first = run(make_chunks(1), FakeGemini(), cache(tmp_path))
    second = run(make_chunks(1), FakeGemini(), cache(tmp_path))
    assert first.cards[0]["tags"] == second.cards[0]["tags"] and "fake_topic" in second.cards[0]["tags"]


def test_a_corrupt_cache_file_is_ignored_and_replaced(tmp_path):
    cch = cache(tmp_path)
    chunks = make_chunks(2)
    run(chunks, FakeGemini(), cch)
    victim = os.path.join(str(tmp_path / "cache" / "chunks"), os.listdir(tmp_path / "cache" / "chunks")[0])
    open(victim, "w").write("{ this is not json")
    fake = FakeGemini()
    res = run(make_chunks(2), fake, cache(tmp_path))
    assert calls(fake) == 1 and res.chunks_done == 2 and res.cache_hits == 1
    json.load(open(victim))                                                      # valid again


# ── when a cached result must NOT be reused ─────────────────────────────────────

def _key(tmp_path, chunk=None, **ctx):
    return cache(tmp_path, **ctx).key(chunk or make_chunks(1)[0])


@pytest.mark.parametrize("ctx", [dict(profile="german"), dict(lang="ru"), dict(deck="Other"), dict(detect_topics=False)])
def test_a_different_setting_is_a_different_key(tmp_path, ctx):
    assert _key(tmp_path, **ctx) != _key(tmp_path)


def test_a_different_text_is_a_different_key(tmp_path):
    a, b = make_chunks(1)[0], make_chunks(1)[0]
    b.text += " one more word"
    assert _key(tmp_path, a) != _key(tmp_path, b)


def test_the_position_in_the_video_is_not_part_of_the_key(tmp_path):
    a, b = make_chunks(1)[0], make_chunks(1)[0]
    b.index, b.total = 7, 9
    assert _key(tmp_path, a) == _key(tmp_path, b)


def test_editing_a_prompt_invalidates_the_cache(tmp_path, monkeypatch):
    before = _key(tmp_path)
    monkeypatch.setattr(anki_gen, "SYSTEM_PROMPT", anki_gen.SYSTEM_PROMPT + "\n13. A brand new rule.")
    assert _key(tmp_path) != before


def test_changing_the_model_invalidates_the_cache(tmp_path, monkeypatch):
    before = _key(tmp_path)
    monkeypatch.setattr(anki_gen, "MODEL", "gemini-9-ultra")
    assert _key(tmp_path) != before


def test_each_profile_fingerprints_its_own_prompts(tmp_path, monkeypatch):
    import profiles
    before = _key(tmp_path, profile="german")
    monkeypatch.setattr(profiles, "GERMAN_SYSTEM_PROMPT", profiles.GERMAN_SYSTEM_PROMPT + " X")
    monkeypatch.setattr(generator, "GERMAN_SYSTEM_PROMPT", profiles.GERMAN_SYSTEM_PROMPT, raising=False)
    assert _key(tmp_path, profile="german") != before


# ── housekeeping ────────────────────────────────────────────────────────────────

def test_old_entries_are_pruned_and_fresh_ones_kept(tmp_path):
    cch = cache(tmp_path)
    run(make_chunks(2), FakeGemini(), cch)
    files = [os.path.join(tmp_path / "cache" / "chunks", f) for f in os.listdir(tmp_path / "cache" / "chunks")]
    old = time.time() - 90 * 86400
    os.utime(files[0], (old, old))
    assert cch.prune(max_age_days=60) == 1 and not os.path.exists(files[0]) and os.path.exists(files[1])


def test_nothing_is_written_without_a_cache(tmp_path):
    res = run(make_chunks(2), FakeGemini(), None)
    assert res.chunks_done == 2 and res.cache_hits == 0 and not os.path.exists(tmp_path / "cache")


# ── audio transcripts (the expensive part of a video without captions) ──────────

def test_a_transcript_round_trips_and_is_specific_to_the_language(tmp_path):
    d = str(tmp_path / "cache")
    entries = [{"text": "hello", "start": 1.0, "duration": 2.0}]
    chunk_cache.save_audio_transcript(d, ID, "ru", "whisper-x", entries, 120.0)
    assert chunk_cache.load_audio_transcript(d, ID, "ru", "whisper-x") == {"entries": entries, "duration": 120.0}
    assert chunk_cache.load_audio_transcript(d, ID, "en", "whisper-x") is None
    assert chunk_cache.load_audio_transcript(d, ID, "ru", "whisper-y") is None
    assert chunk_cache.load_audio_transcript(d, "otherVideo1", "ru", "whisper-x") is None


def test_a_corrupt_transcript_is_ignored(tmp_path):
    d = str(tmp_path / "cache")
    chunk_cache.save_audio_transcript(d, ID, None, "m", [{"text": "x", "start": 0, "duration": 1}], 1.0)
    path = chunk_cache.audio_transcript_path(d, ID, None, "m")
    open(path, "w").write("nope")
    assert chunk_cache.load_audio_transcript(d, ID, None, "m") is None


# ── the whole pipeline ──────────────────────────────────────────────────────────

@pytest.fixture
def env(tmp_path, monkeypatch):
    s, real_time = Sleeper(), time.time
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(usage, "USAGE_FILE", str(tmp_path / "usage.json"))
    for mod in (pipeline, generator):
        monkeypatch.setattr(mod, "time", types.SimpleNamespace(sleep=s, time=real_time))
    cfg = {"gemini_api_key": "k", "groq_api_key": "g", "cache_dir": str(tmp_path / "cache")}
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    gem = FakeGemini()
    monkeypatch.setattr(pipeline, "build_clients", lambda c: [gemini_client(gem)])
    monkeypatch.setattr(generator, "get_transcript_entries",
                        lambda *a, **k: (_ for _ in ()).throw(anki_gen.NoCaptionsError("This video has no captions.")))
    spy = types.SimpleNamespace(audio_calls=0, failed=[], gem=gem, cfg=cfg, tmp=tmp_path)

    def fake_audio(video_id, groq_key, language=None, proxy_url="", on_progress=None):
        spy.audio_calls += 1
        return video_gen.AudioTranscript(entries=block_entries(n_blocks=3), failed_ranges=spy.failed, duration=900.0)

    monkeypatch.setattr(video_gen, "transcribe_youtube_audio", fake_audio)
    return spy


def go(**kw):
    return pipeline.run(pipeline.PipelineConfig(source="youtube", input=f"https://youtu.be/{ID}", deck="D",
                                                run_filter=False, run_quality=False, **kw))


def test_a_rerun_costs_no_audio_transcription_and_no_generation(env):
    first = go()
    gem_calls, audio_calls = calls(env.gem), env.audio_calls
    assert gem_calls == 3 and audio_calls == 1 and first.stages["generation"]["cache_hits"] == 0

    second = go()
    assert calls(env.gem) == gem_calls and env.audio_calls == audio_calls       # nothing was asked of any service
    assert second.stages["generation"]["cache_hits"] == 3 and second.total_final == first.total_final
    assert second.errors == []


def test_cached_chunks_are_not_counted_against_the_quota(env):
    go()
    used = json.load(open(usage.USAGE_FILE))["providers"]["gemini_flash"]["requests"]
    go()
    after = json.load(open(usage.USAGE_FILE))["providers"]
    assert after["gemini_flash"]["requests"] == used and "cache" not in after


def test_a_partly_transcribed_video_is_not_cached(env):
    env.failed = [(600.0, 900.0)]
    go()
    go()
    assert env.audio_calls == 2                                                  # the failed part gets another chance


def test_the_cache_can_be_switched_off(env):
    go()
    go(use_cache=False)
    assert env.audio_calls == 2 and calls(env.gem) == 6


def test_server_forwards_use_cache(monkeypatch):
    import server
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return types.SimpleNamespace(total_final=0, total_raw=0, total_filtered=0, filename="x", report="", duration_sec=0,
                                     stages={}, cards=[], errors=[])

    monkeypatch.setattr(server, "run_youtube", fake_run)
    c = server.app.test_client()
    c.post("/api/generate/youtube", json={"youtube_url": f"https://youtu.be/{ID}"})
    assert seen["use_cache"] is True
    c.post("/api/generate/youtube", json={"youtube_url": f"https://youtu.be/{ID}", "use_cache": False})
    assert seen["use_cache"] is False


def test_ui_has_a_switch_and_sends_it():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, "index.html"), encoding="utf-8").read()
    assert 'id="toggleCache"' in html and "use_cache:" in html and "document.getElementById('toggleCache').checked" in html
