"""
chunk_cache.py — resume a run instead of repeating it.

Free API tiers run out half way through a long video. A re-run used to start from zero: every
chunk that had worked was generated (and paid for) again, and a video without captions was
transcribed again. Now what already succeeded is kept on disk:

  * cache/chunks/<key>.json          the model's cards for ONE chunk (before they are tagged or
                                     given a source time), the topic, and who generated them
  * cache/transcripts/<id>_...json   the Whisper transcript of a video without captions
                                     (only a COMPLETE one: a transcript with failed parts is not kept)

A chunk's key is a hash of everything the result depends on: the chunk's text, profile, card
language, deck, the topic-detection switch, the TEXT of every prompt, and the model. Editing a
prompt therefore invalidates the cache by itself — you can never get cards made by an old prompt.
Chunks that failed are never stored, so a re-run only asks for what is missing.

Plain JSON files, written atomically; a corrupt file is ignored. No imports from the project.
"""

import copy
import hashlib
import json
import os
import tempfile
import time

CACHE_VERSION = 1


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _write_json(path: str, payload: dict) -> None:
    """Atomic: write to a temp file in the same folder, then rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class ChunkCache:
    """Per-chunk results. `context` is everything except the chunk's text that the result depends on."""

    def __init__(self, directory: str, context: dict):
        self.directory = directory
        self.context   = dict(context)
        self.prune()

    # ── keys ────────────────────────────────────────────────────────────────────
    def key(self, chunk) -> str:
        # the position in the video (index/total) is NOT part of the key: the same words give the
        # same cards wherever they sit
        return _sha({"v": CACHE_VERSION, **self.context, "text": chunk.text})

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, "chunks", key + ".json")

    # ── read / write ────────────────────────────────────────────────────────────
    def get(self, chunk):
        """{"cards", "provider", "topic"} for a chunk generated before, else None.
        The cards are a fresh copy: the caller may tag them without touching the cache."""
        data = _read_json(self._path(self.key(chunk)))
        if (not isinstance(data, dict) or data.get("v") != CACHE_VERSION
                or not isinstance(data.get("cards"), list) or not data["cards"]):
            return None
        return {"cards": copy.deepcopy(data["cards"]), "provider": data.get("provider", ""), "topic": data.get("topic", "")}

    def put(self, chunk, cards: list, provider: str, topic: str) -> None:
        try:
            _write_json(self._path(self.key(chunk)), {
                "v": CACHE_VERSION, "cards": copy.deepcopy(cards), "provider": provider, "topic": topic,
                "label": getattr(chunk, "time_label", ""), "created": time.time(),
            })
        except (OSError, TypeError, ValueError):
            pass                                  # a cache must never break a run

    def prune(self, max_age_days: int = 60) -> int:
        """Delete entries older than max_age_days; returns how many."""
        folder = os.path.join(self.directory, "chunks")
        if not os.path.isdir(folder):
            return 0
        limit, removed = time.time() - max_age_days * 86400, 0
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                if os.path.getmtime(path) < limit:
                    os.unlink(path)
                    removed += 1
            except OSError:
                pass
        return removed


# ── audio transcripts ───────────────────────────────────────────────────────────

def audio_transcript_path(directory: str, video_id: str, language, model: str) -> str:
    name = _sha({"v": CACHE_VERSION, "id": video_id, "language": language or "", "model": model})[:24]
    return os.path.join(directory, "transcripts", f"{video_id}_{name}.json")


def load_audio_transcript(directory: str, video_id: str, language, model: str):
    """{"entries", "duration"} of a COMPLETE earlier transcript, else None."""
    data = _read_json(audio_transcript_path(directory, video_id, language, model))
    if (not isinstance(data, dict) or data.get("v") != CACHE_VERSION or not isinstance(data.get("entries"), list)
            or not data["entries"]):
        return None
    return {"entries": data["entries"], "duration": float(data.get("duration") or 0.0)}


def save_audio_transcript(directory: str, video_id: str, language, model: str, entries: list, duration: float) -> None:
    try:
        _write_json(audio_transcript_path(directory, video_id, language, model),
                    {"v": CACHE_VERSION, "entries": entries, "duration": duration, "created": time.time()})
    except (OSError, TypeError, ValueError):
        pass
