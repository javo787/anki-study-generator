"""
A YouTube video WITHOUT usable captions.

Before: the user got ~800 characters of library boilerplate ("please create an issue ...")
with the actual way out buried at the very end. Now: a short message that says what is wrong
and what to do — and, when the captions simply do not exist, the audio is downloaded and
transcribed with Groq Whisper (with timestamps) so the rest of the pipeline, including the
source links, works exactly as with captions.

Offline: no YouTube, no Groq; ffmpeg (really used) for the audio parts.
"""
import inspect
import json
import os
import shutil
import subprocess
import sys
import types

import pytest

import anki_gen
import generator
import pipeline
import usage
import video_gen
from fakes import FakeGemini, Sleeper, block_entries, gemini_client
import youtube_transcript_api._errors as E

ID = "dQw4w9WgXcQ"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


# ── 1. what the user is told ────────────────────────────────────────────────────

def _lib_error(name):
    cls = getattr(E, name)
    values = {"video_id": ID, "requested_language_codes": ["en"], "transcript_data": None,
              "reason": "This video is private", "sub_reasons": [], "http_error": Exception("HTTP 429")}
    args = [values.get(p, ID) for p in list(inspect.signature(cls.__init__).parameters)[1:]]
    return cls(*args)


class _Api:
    err = None
    empty_list = False

    def __init__(self, proxy_config=None):
        pass

    def fetch(self, video_id, languages=("en",)):
        raise _Api.err

    def list(self, video_id):
        if _Api.empty_list:
            class Empty(list):
                def find_transcript(self, codes):
                    raise E.NoTranscriptFound(video_id, codes, None)
            return Empty()
        raise _Api.err


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(anki_gen, "YouTubeTranscriptApi", _Api)
    _Api.empty_list = False
    return _Api


def _message(api, name, proxy=None):
    api.err = _lib_error(name)
    with pytest.raises(RuntimeError) as exc:
        anki_gen.get_transcript_entries(ID, ["en"], proxy)
    return exc.value


@pytest.mark.parametrize("name", ["TranscriptsDisabled", "NoTranscriptFound"])
def test_a_video_without_captions_is_a_distinct_error_with_a_short_message(api, name):
    err = _message(api, name)
    assert isinstance(err, anki_gen.NoCaptionsError)
    assert "no captions" in str(err).lower()
    assert len(str(err)) < 200


def test_a_video_whose_caption_list_is_empty_is_also_no_captions(api):
    api.empty_list = True
    api.err = E.TranscriptsDisabled(ID)
    with pytest.raises(anki_gen.NoCaptionsError):
        anki_gen.get_transcript_entries(ID, ["en"])


@pytest.mark.parametrize("name, must_contain", [
    ("AgeRestricted", ["age-restricted", "Local video"]),
    ("VideoUnavailable", ["unavailable"]),
    ("VideoUnplayable", ["This video is private"]),
    ("RequestBlocked", ["blocking transcript requests", "proxy"]),
    ("IpBlocked", ["blocking transcript requests", "proxy"]),
    ("PoTokenRequired", ["refused", "Local video"]),
    ("YouTubeRequestFailed", ["refused", "HTTP 429", "Local video"]),
])
def test_other_failures_get_a_specific_actionable_message(api, name, must_contain):
    err = _message(api, name)
    assert not isinstance(err, anki_gen.NoCaptionsError)            # these must NOT trigger the audio fallback
    for text in must_contain:
        assert text in str(err)
    assert "create an issue" not in str(err) and "If you are sure" not in str(err)
    assert len(str(err)) < 500


def test_blocked_through_a_proxy_is_not_blamed_on_a_missing_proxy(api):
    err = _message(api, "RequestBlocked", proxy=object())
    assert "even through your proxy" in str(err)


# ── 2. Whisper output -> caption-like entries ───────────────────────────────────

class _Resp:
    def __init__(self, segments=None, text=""):
        self.text = text
        if segments is not None:
            self.segments = segments


class FakeWhisper:
    """Stands in for Groq(): records every call; script(call_no, kwargs) -> response | Exception."""
    def __init__(self, script):
        self.script, self.calls = script, []
        self.audio = types.SimpleNamespace(transcriptions=self)

    def __call__(self, api_key=None):
        return self

    def create(self, **kw):
        kw = {k: v for k, v in kw.items() if k != "file"}
        self.calls.append(kw)
        out = self.script(len(self.calls), kw)
        if isinstance(out, Exception):
            raise out
        return out


def seg(start, end, text, **extra):
    return {"start": start, "end": end, "text": text, **extra}


@pytest.fixture
def audio_env(monkeypatch):
    s = Sleeper()
    monkeypatch.setattr(video_gen.time, "sleep", s)
    return s


@pytest.fixture
def tone(tmp_path):
    def make(seconds):
        path = str(tmp_path / "lecture.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                        "-ac", "1", "-ar", "16000", path], check=True)
        return path
    return make


@needs_ffmpeg
def test_long_audio_is_split_and_every_part_keeps_its_place_in_the_video(monkeypatch, audio_env, tone):
    monkeypatch.setattr(video_gen, "CHUNK_MINUTES", 1)                 # 150 s of audio -> parts of 60 s, 60 s, 30 s
    fake = FakeWhisper(lambda n, kw: _Resp([seg(1.0, 4.0, f" part {n} first"), seg(10.0, 12.5, f" part {n} second")]))
    monkeypatch.setattr(video_gen, "Groq", fake)

    res = video_gen.transcribe_audio_entries(tone(150), "key")

    assert len(fake.calls) == 3
    assert [round(e["start"]) for e in res.entries] == [1, 10, 61, 70, 121, 130]       # offsets 0 / 60 / 120 added
    assert res.entries[0] == {"text": "part 1 first", "start": 1.0, "duration": 3.0}
    assert res.failed_ranges == [] and round(res.duration) == 150
    assert all(kw["response_format"] == "verbose_json" and kw["timestamp_granularities"] == ["segment"] for kw in fake.calls)


@needs_ffmpeg
@pytest.mark.parametrize("seconds, expected_parts", [(120.02, 2), (120.5, 2), (121.5, 3), (60.3, 1)])
def test_a_few_milliseconds_over_a_whole_number_of_parts_does_not_make_a_bogus_extra_part(monkeypatch, audio_env, tone, seconds, expected_parts):
    monkeypatch.setattr(video_gen, "CHUNK_MINUTES", 1)
    fake = FakeWhisper(lambda n, kw: _Resp([seg(0, 1, f"p{n}")]))
    monkeypatch.setattr(video_gen, "Groq", fake)
    res = video_gen.transcribe_audio_entries(tone(seconds), "key")
    assert len(fake.calls) == expected_parts and res.failed_ranges == []


@needs_ffmpeg
def test_spoken_language_is_auto_detected_unless_chosen(monkeypatch, audio_env, tone):
    fake = FakeWhisper(lambda n, kw: _Resp([seg(0, 2, "hi")]))
    monkeypatch.setattr(video_gen, "Groq", fake)
    video_gen.transcribe_audio_entries(tone(20), "key")
    assert "language" not in fake.calls[0]                          # card language must never be sent as "the spoken language"
    video_gen.transcribe_audio_entries(tone(20), "key", language="de")
    assert fake.calls[1]["language"] == "de"


@needs_ffmpeg
def test_whisper_hallucinations_on_silence_and_loops_are_dropped(monkeypatch, audio_env, tone):
    segments = [seg(0, 3, "Real lecture sentence.", avg_logprob=-0.3, no_speech_prob=0.02, compression_ratio=1.4),
                seg(3, 6, "Thanks for watching!", avg_logprob=-1.8, no_speech_prob=0.93, compression_ratio=1.1),     # silence
                seg(6, 9, "the the the the the the the the", avg_logprob=-0.4, no_speech_prob=0.1, compression_ratio=3.1),  # loop
                seg(9, 12, "   ", avg_logprob=-0.2, no_speech_prob=0.0, compression_ratio=1.0),                    # blank
                seg(12, 15, "No stats at all here")]                                                                # keys missing -> keep
    monkeypatch.setattr(video_gen, "Groq", FakeWhisper(lambda n, kw: _Resp(segments)))
    res = video_gen.transcribe_audio_entries(tone(20), "key")
    assert [e["text"] for e in res.entries] == ["Real lecture sentence.", "No stats at all here"]


@needs_ffmpeg
def test_a_response_without_segments_still_gives_a_coarse_entry_per_part(monkeypatch, audio_env, tone):
    monkeypatch.setattr(video_gen, "CHUNK_MINUTES", 1)
    monkeypatch.setattr(video_gen, "Groq", FakeWhisper(lambda n, kw: _Resp(text=f"plain text {n}")))
    res = video_gen.transcribe_audio_entries(tone(90), "key")
    assert [(e["text"], round(e["start"])) for e in res.entries] == [("plain text 1", 0), ("plain text 2", 60)]


@needs_ffmpeg
def test_rate_limit_waits_as_groq_asks_then_continues(monkeypatch, audio_env, tone):
    limited = Exception("Error code: 429 - Rate limit reached for model whisper-large-v3. Please try again in 41.5s.")
    fake = FakeWhisper(lambda n, kw: limited if n <= 2 else _Resp([seg(0, 2, "ok")]))
    monkeypatch.setattr(video_gen, "Groq", fake)
    res = video_gen.transcribe_audio_entries(tone(20), "key")
    assert len(fake.calls) == 3 and audio_env.calls == [pytest.approx(42.5, abs=1.5)] * 2
    assert [e["text"] for e in res.entries] == ["ok"] and res.failed_ranges == []


@needs_ffmpeg
def test_a_part_that_cannot_be_transcribed_is_reported_with_its_time_range(monkeypatch, audio_env, tone):
    monkeypatch.setattr(video_gen, "CHUNK_MINUTES", 1)
    hour = Exception("Error code: 429 - Rate limit ... Please try again in 7m12s.")          # far too long to wait for
    parts = iter([_Resp([seg(0, 2, "one")]), hour, _Resp([seg(0, 2, "three")])])
    monkeypatch.setattr(video_gen, "Groq", FakeWhisper(lambda n, kw: next(parts)))
    res = video_gen.transcribe_audio_entries(tone(150), "key")
    assert [e["text"] for e in res.entries] == ["one", "three"]
    assert res.failed_ranges == [(60.0, 120.0)]
    assert max(audio_env.calls) <= 1                                 # only the 1 s pause between parts: did not sit and wait 7 minutes


@needs_ffmpeg
def test_if_every_part_fails_it_is_an_error_not_an_empty_success(monkeypatch, audio_env, tone):
    monkeypatch.setattr(video_gen, "Groq", FakeWhisper(lambda n, kw: Exception("401 invalid api key")))
    with pytest.raises(RuntimeError, match="could not be transcribed"):
        video_gen.transcribe_audio_entries(tone(20), "key")


# ── 3. downloading the audio ────────────────────────────────────────────────────

def _install_fake_ytdlp(monkeypatch, info, write=True, fail=None):
    class DownloadError(Exception):
        pass

    class YoutubeDL:
        opts = None

        def __init__(self, opts):
            YoutubeDL.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            if fail and not download:
                raise DownloadError(fail)
            return dict(info)

        def process_ie_result(self, info, download=True):
            if fail:
                raise DownloadError(fail)
            if write:
                outdir = os.path.dirname(YoutubeDL.opts["outtmpl"])
                with open(os.path.join(outdir, f"{info['id']}.webm"), "wb") as f:
                    f.write(b"audio")
            return info

    mod, utils = types.ModuleType("yt_dlp"), types.ModuleType("yt_dlp.utils")
    mod.YoutubeDL, utils.DownloadError, mod.utils = YoutubeDL, DownloadError, utils
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", utils)
    return YoutubeDL


def test_audio_is_downloaded_through_yt_dlp(monkeypatch, tmp_path):
    ydl = _install_fake_ytdlp(monkeypatch, {"id": ID, "duration": 3600})
    path = video_gen.download_youtube_audio(ID, str(tmp_path), proxy_url="http://user:pw@proxy:8080")
    assert path == str(tmp_path / f"{ID}.webm") and os.path.exists(path)
    assert ydl.opts["format"] == "bestaudio/best" and ydl.opts["noplaylist"] is True
    assert ydl.opts["proxy"] == "http://user:pw@proxy:8080"


def test_no_proxy_option_when_none_is_configured(monkeypatch, tmp_path):
    ydl = _install_fake_ytdlp(monkeypatch, {"id": ID, "duration": 60})
    video_gen.download_youtube_audio(ID, str(tmp_path))
    assert "proxy" not in ydl.opts


@pytest.mark.parametrize("info, text", [
    ({"id": ID, "is_live": True}, "live"),
    ({"id": ID, "live_status": "is_upcoming"}, "live"),
    ({"id": ID, "duration": 6 * 3600}, "longer than"),
])
def test_live_streams_and_endless_videos_are_refused_before_downloading(monkeypatch, tmp_path, info, text):
    _install_fake_ytdlp(monkeypatch, info)
    with pytest.raises(video_gen.AudioFallbackError, match=text):
        video_gen.download_youtube_audio(ID, str(tmp_path))
    assert os.listdir(tmp_path) == []


def test_download_failure_says_what_to_try(monkeypatch, tmp_path):
    _install_fake_ytdlp(monkeypatch, {"id": ID}, fail="ERROR: Sign in to confirm you're not a bot")
    with pytest.raises(video_gen.AudioFallbackError) as exc:
        video_gen.download_youtube_audio(ID, str(tmp_path))
    msg = str(exc.value)
    assert "Sign in to confirm" in msg and "pip install -U yt-dlp" in msg and "Local video" in msg


def test_missing_yt_dlp_is_explained_not_a_stack_trace(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)                # makes `import yt_dlp` raise ImportError
    with pytest.raises(video_gen.AudioFallbackError, match="pip install yt-dlp"):
        video_gen.download_youtube_audio(ID, str(tmp_path))


@needs_ffmpeg
def test_youtube_audio_end_to_end_with_a_fake_download_and_fake_whisper(monkeypatch, audio_env, tmp_path, tone):
    wav = tone(30)

    def fake_download(video_id, out_dir, proxy_url=""):
        dst = os.path.join(out_dir, f"{video_id}.wav")
        shutil.copy(wav, dst)
        return dst

    monkeypatch.setattr(video_gen, "download_youtube_audio", fake_download)
    monkeypatch.setattr(video_gen, "Groq", FakeWhisper(lambda n, kw: _Resp([seg(2, 5, " hello lecture")])))
    res = video_gen.transcribe_youtube_audio(ID, "key", proxy_url="")
    assert [e["text"] for e in res.entries] == ["hello lecture"]


# ── 4. the pipeline ─────────────────────────────────────────────────────────────

@pytest.fixture
def env(tmp_path, monkeypatch):
    s = Sleeper()
    real_time = __import__("time").time
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(usage, "USAGE_FILE", str(tmp_path / "usage.json"))
    for mod in (pipeline, generator):
        monkeypatch.setattr(mod, "time", types.SimpleNamespace(sleep=s, time=real_time))
    cfg = {"gemini_api_key": "k", "groq_api_key": "g"}
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "build_clients", lambda c: [gemini_client(FakeGemini())])
    monkeypatch.setattr(generator, "get_transcript_entries",
                        lambda *a, **k: (_ for _ in ()).throw(anki_gen.NoCaptionsError("This video has no captions.")))
    spy = types.SimpleNamespace(calls=[])

    def fake_audio(video_id, groq_key, language=None, proxy_url="", on_progress=None):
        spy.calls.append({"video_id": video_id, "key": groq_key, "language": language, "proxy": proxy_url})
        return video_gen.AudioTranscript(entries=spy.entries, failed_ranges=spy.failed, duration=1800.0)

    spy.entries, spy.failed, spy.cfg = block_entries(n_blocks=3), [], cfg
    monkeypatch.setattr(video_gen, "transcribe_youtube_audio", fake_audio)
    return spy


def _run(**kw):
    return pipeline.run(pipeline.PipelineConfig(source="youtube", input=f"https://youtu.be/{ID}", deck="D",
                                                run_filter=False, run_quality=False, **kw))


def test_no_captions_falls_back_to_transcribing_the_audio(env):
    env.cfg["audio_language"] = "ru"
    env.cfg["youtube_proxy_url"] = "http://proxy:1"
    res = _run()
    assert res.total_final == 9 and res.errors == []
    assert res.stages["generation"]["transcript"] == "audio"
    assert env.calls == [{"video_id": ID, "key": "g", "language": "ru", "proxy": "http://proxy:1"}]


def test_cards_from_audio_get_source_links_from_the_whisper_timestamps(env):
    res = _run()
    rows = open(res.filename, encoding="utf-8").read()
    assert f"https://youtu.be/{ID}?t=" in rows


def test_audio_parts_that_failed_are_reported_to_the_user(env):
    env.failed = [(600.0, 1200.0), (3600.0, 4000.0)]
    res = _run()
    assert any("10:00" in e and "20:00" in e and "could not be transcribed" in e for e in res.errors)
    assert any("1:00:00" in e for e in res.errors)


def test_the_fallback_needs_a_groq_key_and_says_so(env):
    env.cfg.pop("groq_api_key")
    with pytest.raises(RuntimeError) as exc:                         # pipeline.run wraps generation errors
        _run()
    msg = str(exc.value)
    assert "no captions" in msg and "Groq key" in msg and "Local video" in msg and env.calls == []
    assert "create an issue" not in msg


def test_the_fallback_can_be_switched_off(env):
    with pytest.raises(RuntimeError) as exc:
        _run(audio_fallback=False)
    assert "switched off" in str(exc.value) and "Local video" in str(exc.value) and env.calls == []


def test_blocked_requests_do_not_trigger_the_audio_fallback(env, monkeypatch):
    monkeypatch.setattr(generator, "get_transcript_entries",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("YouTube is blocking transcript requests")))
    with pytest.raises(RuntimeError, match="blocking"):
        _run()
    assert env.calls == []


def test_silent_audio_is_an_error_not_an_empty_deck(env):
    env.entries = []
    with pytest.raises(RuntimeError, match="No speech"):
        _run()


def test_audio_fallback_failures_reach_the_user_unchanged(env, monkeypatch):
    def boom(*a, **k):
        raise video_gen.AudioFallbackError("Could not download the audio (Sign in to confirm)")
    monkeypatch.setattr(video_gen, "transcribe_youtube_audio", boom)
    with pytest.raises(RuntimeError, match="Sign in to confirm"):
        _run()


def test_normal_videos_never_touch_the_fallback(env, monkeypatch):
    monkeypatch.setattr(generator, "get_transcript_entries", lambda *a, **k: block_entries(n_blocks=3))
    res = _run()
    assert res.stages["generation"]["transcript"] == "captions" and env.calls == []


# ── 5. local video: the card language is not the spoken language ────────────────

@pytest.mark.parametrize("cfg_value, expected", [(None, None), ("", None), ("de", "de")])
def test_local_video_does_not_send_the_card_language_to_whisper(env, monkeypatch, cfg_value, expected):
    seen = {}

    def fake(video_path, groq_api_key, language=None, on_progress=None):
        seen["language"] = language
        return "The choroid plexus produces cerebrospinal fluid. " * 40

    monkeypatch.setattr(video_gen, "transcribe_local_video", fake)
    if cfg_value is not None:
        env.cfg["audio_language"] = cfg_value
    pipeline.run(pipeline.PipelineConfig(source="video", input="lecture.mp4", deck="D", lang="uz",
                                         run_filter=False, run_quality=False))
    assert seen["language"] is expected or seen["language"] == expected


@needs_ffmpeg
def test_transcribe_local_video_omits_language_when_not_given(monkeypatch, audio_env, tone):
    fake = FakeWhisper(lambda n, kw: _Resp(text="hello"))
    monkeypatch.setattr(video_gen, "Groq", fake)
    video_gen.transcribe_local_video(tone(10), "key")
    assert "language" not in fake.calls[0]
    video_gen.transcribe_local_video(tone(10), "key", language="ru")
    assert fake.calls[1]["language"] == "ru"


# ── 6. server + UI ──────────────────────────────────────────────────────────────

def test_server_forwards_audio_fallback_and_returns_the_readable_message(monkeypatch):
    import server
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        raise anki_gen.NoCaptionsError("This video has no captions. Use Local video.")

    monkeypatch.setattr(server, "run_youtube", fake_run)
    client = server.app.test_client()
    r = client.post("/api/generate/youtube", json={"youtube_url": f"https://youtu.be/{ID}"})
    assert seen["audio_fallback"] is True and r.status_code >= 400 and "no captions" in r.get_json()["error"]
    client.post("/api/generate/youtube", json={"youtube_url": f"https://youtu.be/{ID}", "audio_fallback": False})
    assert seen["audio_fallback"] is False


def test_error_messages_are_shown_as_text_not_html():
    import re
    import textwrap
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    html = open(os.path.join(root, "index.html"), encoding="utf-8").read()
    fn = re.search(r"function showError\(msg\) \{.*?\n\}\n", html, re.S).group(0)
    js = textwrap.dedent("""
        const els = {};
        const mk = () => ({ style: {}, children: [], innerHTML: '', textContent: '', className: '',
                            appendChild(c) { this.children.push(c); } });
        global.document = { getElementById: id => (els[id] ||= mk()), createElement: () => mk() };
        %s
        showError('<img src=x onerror=alert(1)> broke');
        const card = els.resultCard;
        console.log(JSON.stringify({ html: card.innerHTML, text: card.children.map(c => c.textContent).join('|') }));
    """) % fn
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert "<img" not in out["html"] and "<img src=x" in out["text"]


def test_webshare_proxy_is_passed_to_yt_dlp_in_the_same_form_as_to_the_captions(env):
    env.cfg.update({"webshare_proxy_username": "bob", "webshare_proxy_password": "pw"})
    _run()
    assert env.calls[0]["proxy"] == "http://bob-rotate:pw@p.webshare.io:80/"
