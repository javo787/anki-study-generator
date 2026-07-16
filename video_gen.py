"""
video_gen.py — Local video/audio → transcript via Groq Whisper
Supports: .flv, .mp4, .mkv, .avi, .mov, .mp3, .wav, .m4a
Requires: pip install groq
ffmpeg must be installed: winget install ffmpeg
"""

import os
import sys
import json
import time
import subprocess
import math
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    sys.exit("❌  Run: pip install groq")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

GROQ_MODEL       = "whisper-large-v3"
MAX_FILE_MB      = 24        # Groq limit is 25MB, keep margin
CHUNK_MINUTES    = 20        # split into 20-minute parts
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
    num_chunks = math.ceil(duration / chunk_sec)

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


def _transcribe_chunk(path: str, client: Groq, language: str = "en") -> str:
    """Send one audio chunk to Groq Whisper and return transcript text."""
    with open(path, "rb") as f:
        response = client.audio.transcriptions.create(
            file     = f,
            model    = GROQ_MODEL,
            language = language,
        )
    return response.text


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def transcribe_local_video(
    video_path:  str,
    groq_api_key: str,
    language:    str = "en",
    on_progress: callable = None,
) -> str:
    """
    Main entry point. Convert + transcribe a local video file.

    Args:
        video_path:    path to .flv, .mp4, .mkv etc.
        groq_api_key:  Groq API key
        language:      language code ("en", "ru", "de")
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
                # Don't stop — continue with other chunks

            # Small pause between requests
            if i < len(chunks):
                time.sleep(1)

        if not transcripts:
            raise RuntimeError("All chunks failed to transcribe.")

        full_transcript = " ".join(transcripts)
        print(f"\n  ✅  Transcript: {len(full_transcript):,} characters")

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

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_gen.py <video_file> [language]")
        print("Example: python video_gen.py lecture.flv en")
        sys.exit(1)

    cfg_path = "config.json"
    if not os.path.exists(cfg_path):
        sys.exit("❌  config.json not found")

    with open(cfg_path) as f:
        cfg = json.load(f)

    groq_key = cfg.get("groq_api_key", "")
    if not groq_key:
        sys.exit("❌  groq_api_key not set in config.json")

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