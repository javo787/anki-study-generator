"""index.html showResult(): partial coverage must be visible. Runs the real function in Node."""
import json
import os
import re
import shutil
import subprocess
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _show_result_source():
    html = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    m = re.search(r"function showResult\(data\) \{.*?\n\}\n", html, re.S)
    assert m, "showResult not found"
    return m.group(0)


def _run(data):
    js = textwrap.dedent("""
        const els = {};
        const mk = () => ({ style: {}, children: [], innerHTML: '', textContent: '', className: '',
                            appendChild(c) { this.children.push(c); }, });
        global.document = { getElementById: id => (els[id] ||= mk()), createElement: () => mk() };
        global.updateProgress = () => {};
        %s
        showResult(%s);
        const w = els.coverageWarn;
        console.log(JSON.stringify({ display: w.style.display, lines: w.children.map(c => c.textContent),
                                     summary: els.stagesSummary.innerHTML }));
    """) % (_show_result_source(), json.dumps(data))
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


BASE = {"count": 40, "raw": 40, "filtered": 40, "duration": 5, "preview": [],
        "stages": {"generation": {"cards": 40, "chunks": 4, "chunks_total": 9}}}


def test_partial_coverage_is_shown_to_the_user():
    got = _run({**BASE, "errors": ["Only 4 of 9 parts of the source produced cards (44% coverage) — MISSING",
                                   "Chunk 2 (05:03 – 10:03): no cards generated"]})
    assert got["display"] == "block"
    assert got["lines"][0].startswith("⚠️ Only 4 of 9")
    assert got["lines"][1].startswith("• Chunk 2")
    assert "4 of 9 chunks" in got["summary"]


def test_clean_run_shows_no_warning():
    got = _run({**BASE, "errors": []})
    assert got["display"] == "none" and got["lines"] == []
