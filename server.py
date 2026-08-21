"""
Anki Card Generator — Web Server
Run:  python server.py
Open: http://localhost:5000

Requires: pip install flask flask-cors
"""

import os
import sys
import uuid
import threading
import webbrowser
from datetime import datetime

# ── IMPORTS ────────────────────────────────────────────────────────────────────

try:
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
except ImportError:
    sys.exit("❌  Run: pip install flask flask-cors")

from pipeline import run_youtube, run_pdf, run_video, PipelineConfig, save_cards
from usage    import get_all_usage, reset_usage, print_usage
from config   import load_config, save_config, mask_key

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "output"
PORT        = 5000

app  = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 2048 * 1024 * 1024  # 2 GB ceiling for lecture videos

# Progress state per session
_progress_store = {}

# ── UPLOADS ────────────────────────────────────────────────────────────────────

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_VIDEO_EXTS = {".flv", ".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".m4a", ".ogg", ".webm"}
ALLOWED_PDF_EXTS   = {".pdf"}


def _is_temp_upload(path: str) -> bool:
    """True if this path lives inside our uploads folder (safe to delete after use)."""
    try:
        return os.path.dirname(os.path.abspath(path)) == UPLOAD_DIR
    except Exception:
        return False


def _handle_upload(allowed_exts: set):
    """
    Shared real-file-upload handler for both video and PDF drop-zones.
    Browsers never expose the true local path of a picked file (security
    restriction), so we accept the file's bytes here, save it under a random
    name in UPLOAD_DIR, and hand back the real server-side path that the
    generate routes can use.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"Could not save upload: {e}"}), 500

    return jsonify({
        "ok":            True,
        "path":          os.path.abspath(save_path),
        "original_name": secure_filename(file.filename),
        "size_mb":       round(os.path.getsize(save_path) / (1024 * 1024), 1),
    })


@app.route("/api/upload/video", methods=["POST"])
def route_upload_video():
    return _handle_upload(ALLOWED_VIDEO_EXTS)


@app.route("/api/upload/pdf", methods=["POST"])
def route_upload_pdf():
    return _handle_upload(ALLOWED_PDF_EXTS)

# ── CONFIG ─────────────────────────────────────────────────────────────────────

# ── ROUTES — CONFIG ────────────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def route_get_config():
    cfg  = load_config()
    safe = {}
    for k, v in cfg.items():
        safe[k] = mask_key(v) if "key" in k and v else v
    safe["keys_set"] = {
        "gemini_flash": bool(cfg.get("gemini_api_key")),
        "gemini_pro":   bool(cfg.get("gemini_pro_key")),
        "groq":         bool(cfg.get("groq_api_key")),
        "openrouter":   bool(cfg.get("openrouter_api_key")),
        "cerebras":     bool(cfg.get("cerebras_api_key")),
        "nvidia":       bool(cfg.get("nvidia_api_key")),
    }
    return jsonify(safe)


@app.route("/api/config", methods=["POST"])
def route_post_config():
    data = request.json or {}
    cfg  = load_config()
    fields = [
        "gemini_api_key", "gemini_pro_key", "groq_api_key",
        "openrouter_api_key", "cerebras_api_key", "nvidia_api_key",
        "default_lang", "default_format", "chunk_size",
    ]
    for field in fields:
        if field in data and data[field]:
            cfg[field] = data[field]
    save_config(cfg)
    return jsonify({"ok": True})

# ── ROUTES — GENERATE ──────────────────────────────────────────────────────────

@app.route("/api/generate/youtube", methods=["POST"])
def route_youtube():
    data = request.json or {}
    url  = data.get("youtube_url", "").strip()
    deck = data.get("deck", "My Deck").strip()
    lang = data.get("lang", "en")
    session_id = data.get("session_id", "default")

    if not url:
        return jsonify({"error": "YouTube URL is required."}), 400

    def on_progress(info: dict):
        _progress_store[session_id] = info

    try:
        result = run_youtube(
            url         = url,
            deck        = deck,
            lang        = lang,
            run_filter  = data.get("run_filter",  True),
            run_quality = data.get("run_quality", True),
            on_progress = on_progress,
        )
        return jsonify({
            "ok":       True,
            "count":    result.total_final,
            "raw":      result.total_raw,
            "filtered": result.total_filtered,
            "filename": os.path.basename(result.filename),
            "duration": round(result.duration_sec, 1),
            "stages":   result.stages,
            "preview":  result.cards[:3],
            "errors":   result.errors,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate/pdf", methods=["POST"])
def route_pdf():
    data      = request.json or {}
    pdf_path  = data.get("pdf_path", "").strip().strip('"')
    deck      = data.get("deck", "My Deck").strip()
    lang      = data.get("lang", "en")
    page_from = int(data.get("page_from", 1))
    page_to   = int(data.get("page_to", 99999))
    session_id = data.get("session_id", "default")

    if not pdf_path:
        return jsonify({"error": "PDF path is required."}), 400
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"File not found: {pdf_path}"}), 400

    def on_progress(info: dict):
        _progress_store[session_id] = info

    cleanup_after = _is_temp_upload(pdf_path)

    try:
        result = run_pdf(
            path        = pdf_path,
            deck        = deck,
            lang        = lang,
            page_from   = page_from,
            page_to     = page_to,
            run_filter  = data.get("run_filter",  True),
            run_quality = data.get("run_quality", True),
            on_progress = on_progress,
        )
        return jsonify({
            "ok":       True,
            "count":    result.total_final,
            "raw":      result.total_raw,
            "filtered": result.total_filtered,
            "filename": os.path.basename(result.filename),
            "duration": round(result.duration_sec, 1),
            "stages":   result.stages,
            "preview":  result.cards[:3],
            "errors":   result.errors,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cleanup_after and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError:
                pass


@app.route("/api/generate/video", methods=["POST"])
def route_video():
    data       = request.json or {}
    video_path = data.get("video_path", "").strip().strip('"')
    deck       = data.get("deck", "My Deck").strip()
    lang       = data.get("lang", "en")
    session_id = data.get("session_id", "default")

    if not video_path:
        return jsonify({"error": "Video file path is required."}), 400
    if not os.path.exists(video_path):
        return jsonify({"error": f"File not found: {video_path}"}), 400

    def on_progress(info: dict):
        _progress_store[session_id] = info

    cleanup_after = _is_temp_upload(video_path)

    try:
        result = run_video(
            path        = video_path,
            deck        = deck,
            lang        = lang,
            run_filter  = data.get("run_filter",  True),
            run_quality = data.get("run_quality", True),
            on_progress = on_progress,
        )
        return jsonify({
            "ok":       True,
            "count":    result.total_final,
            "raw":      result.total_raw,
            "filtered": result.total_filtered,
            "filename": os.path.basename(result.filename),
            "duration": round(result.duration_sec, 1),
            "stages":   result.stages,
            "preview":  result.cards[:3],
            "errors":   result.errors,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if cleanup_after and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass

# ── ROUTES — PROGRESS ──────────────────────────────────────────────────────────

@app.route("/api/progress/<session_id>")
def route_progress(session_id):
    """Poll endpoint for generation progress."""
    info = _progress_store.get(session_id, {"stage": "idle", "pct": 0})
    return jsonify(info)

# ── ROUTES — USAGE ─────────────────────────────────────────────────────────────

@app.route("/api/usage")
def route_usage():
    return jsonify(get_all_usage())


@app.route("/api/usage/reset", methods=["POST"])
def route_reset_usage():
    data     = request.json or {}
    provider = data.get("provider", None)
    reset_usage(provider)
    return jsonify({"ok": True})

# ── ROUTES — FILES ─────────────────────────────────────────────────────────────

@app.route("/api/files")
def route_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = []
    for f in sorted(os.listdir(OUTPUT_DIR), reverse=True):
        if f.endswith(".txt"):
            path = os.path.join(OUTPUT_DIR, f)
            files.append({
                "name":     f,
                "size":     os.path.getsize(path),
                "modified": datetime.fromtimestamp(
                    os.path.getmtime(path)
                ).strftime("%Y-%m-%d %H:%M"),
            })
    return jsonify(files)


@app.route("/api/download/<filename>")
def route_download(filename):
    full_path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if os.path.exists(full_path):
        return send_file(full_path, as_attachment=True)
    return jsonify({"error": "File not found."}), 404


@app.route("/")
def route_index():
    return send_file("index.html")

# ── START ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n🧠  Anki Generator Pro — Web UI")
    print(f"   http://localhost:{PORT}\n")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(debug=False, port=PORT, use_reloader=False)