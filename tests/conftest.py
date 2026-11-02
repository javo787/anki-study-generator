"""Shared pytest setup: make the project root importable from tests/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """The resume cache is on by default and keyed by the chunk's text: without this, tests that
    happen to use the same text would silently reuse each other's results (and write into the repo)."""
    import pipeline
    monkeypatch.setattr(pipeline, "CACHE_DIR", str(tmp_path / "default_cache"))
