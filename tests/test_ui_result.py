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


# ── content-type picker ─────────────────────────────────────────────────────────

def _run_chip(group, value):
    html = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    fn = re.search(r"function selectChip\(group, el, value\) \{.*?\n\}\n", html, re.S).group(0)
    js = textwrap.dedent("""
        const els = {};
        const mkEl = () => ({ style: {}, classList: { removed: 0, added: [], remove() { this.removed++; }, add(c) { this.added.push(c); } } });
        global.document = { getElementById: id => (els[id] ||= mkEl()),
                            querySelectorAll: sel => [mkEl(), mkEl()] };
        let currentLang = 'en', currentProfile = 'medical';
        %s
        const el = mkEl();
        selectChip(%s, el, %s);
        console.log(JSON.stringify({ lang: currentLang, profile: currentProfile,
                                     hint: (els.profileHint || {style: {}}).style.display, active: el.classList.added }));
    """) % (fn, json.dumps(group), json.dumps(value))
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


def test_choosing_german_sets_the_profile_and_shows_the_hint():
    got = _run_chip("profile", "german")
    assert got == {"lang": "en", "profile": "german", "hint": "block", "active": ["active"]}


def test_choosing_medical_hides_the_hint_and_the_language_chips_still_work():
    assert _run_chip("profile", "medical")["hint"] == "none"
    got = _run_chip("lang", "uz")
    assert got["lang"] == "uz" and got["profile"] == "medical"


# ── change report link ──────────────────────────────────────────────────────────

def _run_with_report(data):
    js = textwrap.dedent("""
        const els = {};
        const mk = () => ({ style: {}, children: [], innerHTML: '', textContent: '', className: '', href: '',
                            appendChild(c) { this.children.push(c); } });
        global.document = { getElementById: id => (els[id] ||= mk()), createElement: () => mk() };
        global.updateProgress = () => {};
        %s
        showResult(%s);
        const w = els.coverageWarn;
        const links = [];
        const walk = n => { if (n.href) links.push({ href: n.href, text: n.textContent }); (n.children || []).forEach(walk); };
        walk(w);
        console.log(JSON.stringify({ display: w.style.display, links }));
    """) % (_show_result_source(), json.dumps(data))
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


def test_the_change_report_is_offered_as_a_link_even_without_other_warnings():
    got = _run_with_report({**BASE, "errors": [], "report": "anki_Neuro_2026_changes.md"})
    assert got["display"] == "block"
    assert got["links"] == [{"href": "/api/download/anki_Neuro_2026_changes.md", "text": "📝 Open the change report (before / after of every card the AI changed)"}]


def test_the_report_name_is_url_encoded():
    got = _run_with_report({**BASE, "errors": [], "report": "a b&c?.md"})
    assert got["links"][0]["href"] == "/api/download/a%20b%26c%3F.md"


def test_no_report_no_link():
    got = _run_with_report({**BASE, "errors": [], "report": ""})
    assert got["links"] == [] and got["display"] == "none"
