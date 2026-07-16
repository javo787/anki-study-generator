"""
config.py — Single shared config module for the whole project.

Every other file (main.py, pipeline.py, server.py, usage.py, video_gen.py,
test_gen.py) imports load_config()/save_config()/DEFAULT_CONFIG from HERE
instead of each keeping its own copy. Before this, the same load_config()
logic was duplicated in six different files — this is the one source of
truth now.

config.json itself holds your real API keys locally and is gitignored —
it never gets committed. config.example.json (safe, no secrets) is the
template that IS committed, so the project stays clonable by anyone.
"""

import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "gemini_api_key":     "",
    "gemini_pro_key":     "",
    "groq_api_key":       "",
    "openrouter_api_key": "",
    "cerebras_api_key":   "",
    "nvidia_api_key":     "",   # was used by pipeline.py's client builder but
                                # never had a default here or a settings UI
                                # field — added so it's actually manageable.
    "default_lang":       "en",
    "default_format":     "both",
    "chunk_size":         6000,
}


def load_config() -> dict:
    """Load config.json, filling in any missing fields with defaults."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    """Write the given config dict back to config.json."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def mask_key(key: str) -> str:
    """Show only the last 4 characters of a secret — safe for display in UIs/logs."""
    if not key:
        return "(not set)"
    if len(key) <= 4:
        return "*" * len(key)
    return "*" * (len(key) - 4) + key[-4:]
