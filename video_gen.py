"""
video_gen.py — audio → transcript via Groq Whisper

Two jobs:
  * transcribe_local_video():    a video file on your disk -> plain text
  * transcribe_youtube_audio():  a YouTube video WITHOUT captions -> caption-like entries
                                 with timestamps (yt-dlp downloads the audio), so the rest
                                 of the pipeline — chunking, source links — works as with captions

Supports: .flv, .mp4, .mkv, .avi, .mov, .mp3, .wav, .m4a
Requires: pip install groq   (and, for YouTube audio: pip install yt-dlp)
ffmpeg must be installed: winget install ffmpeg
"""

import os
import sys
import json
import time
import subprocess
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    sys.exit("❌  Run: pip install groq")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

GROQ_MODEL       = "whisper-large-v3"
MAX_FILE_MB      = 24        # Groq limit is 25MB, keep margin
CHUNK_MINUTES    = 20        # split into 20-minute parts
MIN_TAIL_SECONDS    = 1.0    # shorter audio tails are not worth a part of their own
MAX_AUDIO_MINUTES   = 300    # refuse longer YouTube videos (hours of audio would use up the Whisper quota)
MAX_RATE_LIMIT_WAIT = 90     # seconds we are willing to wait when Groq says "try again in N s"
# Whisper's own rules for "this segment is not speech" (openai/whisper transcribe.py defaults)
NO_SPEECH_THRESHOLD    = 0.6
LOGPROB_THRESHOLD      = -1.0
COMPRESSION_THRESHOLD  = 2.4
SUPPORTED_EXTS   = {
    ".flv", ".mp4", ".mkv", ".avi", ".mov",
    ".mp3", ".wav", ".m4a", ".ogg", ".webm"
}

# ── HELPERS ────────────────────────────────────────────────────────────────────

def _check_ffmpeg():
    """Verify ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg not found. Install it: winget install ffmpeg\n"
            "Then restart your terminal."
        )


def _get_duration_seconds(path: str) -> float:
    """Get video/audio duration using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", path
        ],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def _convert_to_mp3(input_path: str, output_path: str):
    """Convert any video/audio to mp3 using ffmpeg."""
    print(f"  🔄  Converting to MP3...")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vn",                    # no video
            "-acodec", "libmp3lame",
            "-ar", "16000",           # 16kHz — enough for speech
            "-ac", "1",               # mono
            "-b:a", "48k",            # low bitrate — smaller file
            output_path
        ],
        capture_output=True, check=True
    )


def _split_audio(input_path: str, output_dir: str, chunk_minutes: int) -> list[str]:
    """
    Split audio into chunks of chunk_minutes each.
    Returns list of chunk file paths.
    """
    duration = _get_duration_seconds(input_path)
    chunk_sec = chunk_minutes * 60
    # A tail shorter than MIN_TAIL_SECONDS is dropped: it cannot hold speech, and ffmpeg
    # writes an invalid file for it that Whisper rejects. (Audio that is exactly N parts long
    # is a few ms over N*chunk_sec — it used to get a bogus extra part.)
    num_chunks = max(1, math.ceil((duration - MIN_TAIL_SECONDS) / chunk_sec))

    print(f"  ✂️   Splitting into {num_chunks} parts ({chunk_minutes} min each)...")

    chunks = []
    for i in range(num_chunks):
        start   = i * chunk_sec
        out_path = os.path.join(output_dir, f"chunk_{i+1:02d}.mp3")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ss", str(start),
                "-t",  str(chunk_sec),
                "-acodec", "copy",
                out_path
            ],
            capture_output=True, check=True
        )
        chunks.append(out_path)

    return chunks


def _file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def _transcribe_chunk(path: str, client: Groq, language: str = None) -> str:
    """Send one audio chunk to Groq Whisper and return transcript text.

    `language` is the language SPOKEN in the audio. Leave it None to let Whisper detect it:
    it used to be filled with the *card* language, so asking for Uzbek cards from an English
    lecture told Whisper the lecture was Uzbek."""
    kwargs = {"model": GROQ_MODEL}
    if language:
        kwargs["language"] = language
    with open(path, "rb") as f:
        response = client.audio.transcriptions.create(file=f, **kwargs)
    return response.text


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def transcribe_local_video(
    video_path:  str,
    groq_api_key: str,
    language:    str = None,
    on_progress: callable = None,
) -> str:
    """
    Main entry point. Convert + transcribe a local video file.

    Args:
        video_path:    path to .flv, .mp4, .mkv etc.
        groq_api_key:  Groq API key
        language:      language SPOKEN in the video ("en", "ru", "de"); None = detect automatically
        on_progress:   optional callback(step, total, message)

    Returns:
        full transcript as plain text
    """
    _check_ffmpeg()

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {video_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported format: {ext}. Supported: {SUPPORTED_EXTS}")

    client   = Groq(api_key=groq_api_key)
    tmp_dir  = str(path.parent / f"_anki_tmp_{path.stem}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # ── Step 1: Convert to MP3 ─────────────────────────────────────────────
        mp3_path = os.path.join(tmp_dir, "audio.mp3")
        _convert_to_mp3(str(path), mp3_path)
        size_mb = _file_mb(mp3_path)
        print(f"  ✅  Converted: {size_mb:.1f} MB")

        if on_progress:
            on_progress(1, 3, f"Converted to MP3 ({size_mb:.1f} MB)")

        # ── Step 2: Split if too large ─────────────────────────────────────────
        if size_mb > MAX_FILE_MB:
            chunks = _split_audio(mp3_path, tmp_dir, CHUNK_MINUTES)
        else:
            chunks = [mp3_path]

        print(f"  📦  {len(chunks)} audio chunk(s) to transcribe")

        # ── Step 3: Transcribe each chunk ──────────────────────────────────────
        transcripts = []
        failed_chunks = []
        for i, chunk_path in enumerate(chunks, 1):
            chunk_mb = _file_mb(chunk_path)
            print(f"  🎙️   Transcribing chunk {i}/{len(chunks)} ({chunk_mb:.1f} MB)...", end=" ", flush=True)

            if on_progress:
                on_progress(i + 1, len(chunks) + 2, f"Transcribing part {i}/{len(chunks)}")

            try:
                text = _transcribe_chunk(chunk_path, client, language)
                transcripts.append(text)
                print(f"{len(text.split())} words")
            except Exception as e:
                print(f"failed ({e})")
                failed_chunks.append(i)
                # Don't stop — other chunks may still succeed, but we surface
                # exactly which parts of the lecture are missing at the end.

            # Small pause between requests
            if i < len(chunks):
                time.sleep(1)

        if not transcripts:
            raise RuntimeError("All chunks failed to transcribe.")

        if failed_chunks:
            print(f"\n  ⚠️   {len(failed_chunks)}/{len(chunks)} chunk(s) failed and are MISSING "
                  f"from the transcript: parts {failed_chunks}")

        full_transcript = " ".join(transcripts)
        print(f"\n  ✅  Transcript: {len(full_transcript):,} characters"
              + (f" ({len(failed_chunks)} part(s) missing)" if failed_chunks else ""))

        if on_progress:
            on_progress(len(chunks) + 2, len(chunks) + 2, "Transcription complete")

        return full_transcript

    finally:
        # ── Cleanup temp files ─────────────────────────────────────────────────
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            print(f"  🗑️   Cleaned up temp files")


# ── DEBUG ──────────────────────────────────────────────────────────────────────

# ── YOUTUBE VIDEOS WITHOUT CAPTIONS ────────────────────────────────────────────

class AudioFallbackError(RuntimeError):
    """The audio of a YouTube video could not be downloaded or transcribed."""


@dataclass
class AudioTranscript:
    """Speech-to-text result shaped like YouTube captions, so chunk_youtube() takes it as is."""
    entries:       list                                        # {"text", "start", "duration"} (seconds)
    failed_ranges: list = field(default_factory=list)          # [(start, end)] parts that could not be transcribed
    duration:      float = 0.0


def _groq_retry_seconds(err: str):
    """Seconds Groq asked us to wait ('Please try again in 7m12s' / '41.5s' / '1h2m3s'), or None."""
    m = re.search(r"try again in ((?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)", err, re.IGNORECASE)
    if not m or not m.group(1):
        return None
    seconds = 0.0
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", m.group(1)):
        seconds += float(number) * {"h": 3600, "m": 60, "s": 1}[unit]
    return seconds


def _get(seg, key, default=None):
    """A Whisper segment is a dict in the SDK today; accept objects too."""
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _keep_segment(seg) -> bool:
    """Drop what Whisper itself would call 'not speech': silence/music it fills with 'Thanks for
    watching' (no_speech_prob high AND logprob low) and repetition loops (compression ratio)."""
    if not " ".join(str(_get(seg, "text", "") or "").split()):
        return False
    no_speech = _get(seg, "no_speech_prob")
    logprob   = _get(seg, "avg_logprob")
    if no_speech is not None and logprob is not None and no_speech > NO_SPEECH_THRESHOLD and logprob < LOGPROB_THRESHOLD:
        return False
    ratio = _get(seg, "compression_ratio")
    if ratio is not None and ratio > COMPRESSION_THRESHOLD:
        return False
    return True


def _segments_to_entries(response, offset: float, part_seconds: float) -> list:
    """One Whisper response (for one audio part) -> caption-like entries at the right place in the video."""
    segments = _get(response, "segments") or []
    if not segments:
        text = " ".join(str(_get(response, "text", "") or "").split())
        if not text:
            return []
        print("  ⚠️   No segment timestamps in the response — this part gets one coarse entry")
        return [{"text": text, "start": float(offset), "duration": float(part_seconds)}]
    entries = []
    for seg in segments:
        if not _keep_segment(seg):
            continue
        start = float(_get(seg, "start", 0.0) or 0.0)
        end   = float(_get(seg, "end", start) or start)
        entries.append({"text": " ".join(str(_get(seg, "text")).split()),
                        "start": offset + start, "duration": max(0.1, end - start)})
    return entries


def _transcribe_part(path: str, client, language):
    """Whisper for one part, with timestamps. Waits as Groq asks on a rate limit (up to
    MAX_RATE_LIMIT_WAIT seconds, 3 attempts); anything else is raised to the caller."""
    kwargs = {"model": GROQ_MODEL, "response_format": "verbose_json", "timestamp_granularities": ["segment"]}
    if language:
        kwargs["language"] = language
    for attempt in range(1, 4):
        try:
            with open(path, "rb") as f:
                return client.audio.transcriptions.create(file=f, **kwargs)
        except Exception as e:
            err = str(e)
            transient = "429" in err or "503" in err or "rate limit" in err.lower()
            wait = _groq_retry_seconds(err)
            if not transient or attempt == 3 or (wait is not None and wait > MAX_RATE_LIMIT_WAIT):
                raise
            pause = wait + 1 if wait is not None else attempt * 20
            print(f"  ⏳  Groq rate limit — waiting {pause:.0f}s (attempt {attempt}/3)...")
            time.sleep(pause)


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcribe_audio_entries(
    audio_path:   str,
    groq_api_key: str,
    language:     str = None,
    on_progress:  callable = None,
) -> AudioTranscript:
    """
    Transcribe an audio/video file into caption-like entries WITH timestamps.

    The file is converted, cut into CHUNK_MINUTES parts (Groq's upload limit) and each part is
    transcribed; its segment times are shifted by the part's offset. A part that fails does not
    sink the run — its time range is returned in `failed_ranges` so the caller can tell the
    user which part of the video is missing. If EVERY part fails that is an error.

    language: language SPOKEN in the audio; None = Whisper detects it.
    """
    _check_ffmpeg()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    client  = Groq(api_key=groq_api_key)
    tmp_dir = tempfile.mkdtemp(prefix="_anki_audio_")
    try:
        mp3 = os.path.join(tmp_dir, "audio.mp3")
        _convert_to_mp3(audio_path, mp3)
        total     = _get_duration_seconds(mp3)
        chunk_sec = CHUNK_MINUTES * 60
        parts     = _split_audio(mp3, tmp_dir, CHUNK_MINUTES) if total > chunk_sec else [mp3]
        print(f"  📦  {len(parts)} audio part(s), {total / 60:.0f} min")

        entries, failed, last_error = [], [], None
        for i, part in enumerate(parts):
            offset   = i * chunk_sec
            part_len = min(chunk_sec, max(0.0, total - offset))
            if on_progress:
                on_progress(i, len(parts), f"Transcribing audio {i + 1}/{len(parts)}")
            print(f"  🎙️   Transcribing part {i + 1}/{len(parts)} ({_fmt(offset)} – {_fmt(offset + part_len)})...", end=" ", flush=True)
            try:
                response = _transcribe_part(part, client, language)
                got = _segments_to_entries(response, offset, part_len)
                entries.extend(got)
                print(f"{len(got)} segments")
            except Exception as e:
                last_error = e
                failed.append((float(offset), float(offset + part_len)))
                print(f"FAILED ({e.__class__.__name__})")
            if i < len(parts) - 1:
                time.sleep(1)

        if len(failed) == len(parts):
            raise AudioFallbackError(
                f"The audio could not be transcribed (all {len(parts)} part(s) failed: {last_error}). "
                "Check the Groq key and quota, or try again later.")
        if on_progress:
            on_progress(len(parts), len(parts), "Transcription done")
        return AudioTranscript(entries=entries, failed_ranges=failed, duration=total)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_youtube_audio(video_id: str, out_dir: str, proxy_url: str = "") -> str:
    """
    Download the audio track of a YouTube video with yt-dlp; returns the file path.
    Refuses live streams and videos longer than MAX_AUDIO_MINUTES before downloading anything.
    """
    tip = ("Update yt-dlp (pip install -U yt-dlp), add a proxy in Settings -> YouTube proxy, "
           "or download the video yourself and use the 'Local video' tab.")
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError:
        raise AudioFallbackError(
            "Automatic audio transcription needs yt-dlp. Run: pip install yt-dlp (and install ffmpeg). "
            "Or download the video yourself and use the 'Local video' tab.")

    opts = {
        "format":       "bestaudio/best",
        "outtmpl":      os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist":   True,
        "quiet":        True,
        "no_warnings":  True,
        "retries":      3,
        "socket_timeout": 30,
    }
    if proxy_url:
        opts["proxy"] = proxy_url

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
                raise AudioFallbackError("This is a live stream (or one that has not started) — there is no finished audio to transcribe.")
            if (info.get("duration") or 0) > MAX_AUDIO_MINUTES * 60:
                raise AudioFallbackError(f"This video is longer than {MAX_AUDIO_MINUTES // 60} hours — too long to transcribe automatically.")
            print(f"  ⬇️   Downloading audio ({(info.get('duration') or 0) / 60:.0f} min)...")
            ydl.process_ie_result(info, download=True)
    except DownloadError as e:
        cause = " ".join(str(e).replace("ERROR:", "").split())[:200]
        raise AudioFallbackError(f"Could not download the audio ({cause}). {tip}") from e

    files = sorted(f for f in os.listdir(out_dir) if f.startswith(video_id) and not f.endswith((".part", ".ytdl")))
    if not files:
        raise AudioFallbackError(f"The download produced no audio file. {tip}")
    return os.path.join(out_dir, files[0])


def transcribe_youtube_audio(
    video_id:     str,
    groq_api_key: str,
    language:     str = None,
    proxy_url:    str = "",
    on_progress:  callable = None,
) -> AudioTranscript:
    """YouTube video id -> AudioTranscript (download the audio, transcribe it with timestamps)."""
    tmp_dir = tempfile.mkdtemp(prefix="_anki_yt_")
    try:
        if on_progress:
            on_progress(0, 1, "Downloading audio")
        path = download_youtube_audio(video_id, tmp_dir, proxy_url)
        return transcribe_audio_entries(path, groq_api_key, language, on_progress)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_gen.py <video_file> [language]")
        print("Example: python video_gen.py lecture.flv en")
        sys.exit(1)

    from config import load_config

    cfg = load_config()
    groq_key = cfg.get("groq_api_key", "")
    if not groq_key:
        sys.exit("❌  groq_api_key not set. Run: python main.py → Settings, or set it in the web UI.")

    video  = sys.argv[1]
    lang   = sys.argv[2] if len(sys.argv) > 2 else "en"

    print(f"\n🎬  Transcribing: {video}\n")
    transcript = transcribe_local_video(video, groq_key, lang)

    out_dir  = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / (Path(video).stem + "_transcript.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"\n💾  Saved: {out_file}")