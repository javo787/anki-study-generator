"""
uz_transliterate.py — Deterministic Uzbek Latin → Cyrillic transliteration.

Why this exists: Gemini/Groq are instructed (see anki_gen.LANGUAGE / the
{language_instruction} block) to write Uzbek card text directly in Cyrillic.
Models don't always fully comply — Latin is the dominant script in their
training data for modern Uzbek, so weaker fallback models especially tend to
default back to it. This module is the deterministic SAFETY NET used by
format_pass.py: it guarantees Cyrillic output even if the model ignores the
instruction, the same way format_pass.py's _validate_and_repair() guarantees
real {{c1::}} cloze syntax even if the model ignores THAT instruction.

Mapping follows the standard 1995 Latin <-> 1940 Cyrillic correspondence
(the alphabet still taught/used across Uzbekistan, Tajikistan, Kyrgyzstan,
Kazakhstan). Validated against the Uzbek UDHR Article 1 parallel text:

  Latin:    Barcha odamlar erkin, qadr-qimmat va huquqlarda teng boʻlib
            tugʻiladilar.
  Cyrillic: Барча одамлар эркин, қадр-қиммат ва ҳуқуқларда тенг бўлиб
            туғиладилар.

Known limitations (this is a best-effort safety net, not a linguistic
parser):
  - Bare Latin "c" and "w" are left untouched. Neither is a real Uzbek Latin
    letter (only "ch", "ts" are), so this is usually correct — and it has
    the convenient side effect of leaving Anki cloze markers ({{c1::...}})
    completely alone, since nothing in {{c1:: or }} is otherwise a letter.
  - "e" is rendered as "э" only at the very start of a word and "е"
    everywhere else. Real orthography also uses "е" after a vowel
    mid-word — a rarer case this simplifies away.
  - "ts" -> "ц" is a one-way heuristic for loanwords (qarants ~ карантин
    etc.); it can't always be un-ambiguously reversed to "ц" vs plain "с".
  - HTML tags (<b>, <br>, ...) are detected and passed through untouched.
"""

import re

# ── APOSTROPHE VARIANTS ────────────────────────────────────────────────────
# Real Uzbek Latin text is supposed to use U+02BB (ʻ MODIFIER LETTER TURNED
# COMMA) for oʻ/gʻ, but in practice — LLM output included — it's almost
# always typed as a plain apostrophe. We treat every common look-alike the
# same way so "o'zbek", "o‘zbek" and "oʻzbek" all convert identically.
_APOS = "ʻʼ\u2018\u2019'"          # ʻ ʼ ‘ ’ '
_APOS_CLASS = "[" + _APOS + "]"

_SINGLE = {
    "a": "а", "b": "б", "d": "д", "f": "ф", "g": "г", "h": "ҳ", "i": "и",
    "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "x": "х",
    "y": "й", "z": "з",
}
_DIGRAPH = {
    "sh": "ш", "ch": "ч", "ts": "ц", "yo": "ё", "yu": "ю", "ya": "я", "ye": "е",
}

_TOKEN_RE = re.compile(
    r"s" + _APOS_CLASS + r"h"      # sʼh disambiguator -> с + ҳ, e.g. Isʼhoq
    r"|o" + _APOS_CLASS +          # oʻ / o' / o' -> ў
    r"|g" + _APOS_CLASS +          # gʻ / g' / g' -> ғ
    r"|sh|ch|ts|ng|yo|yu|ya|ye"    # remaining digraphs
    r"|" + _APOS_CLASS +           # bare tutuq belgisi -> ъ
    r"|[a-zA-Z]",
    re.IGNORECASE,
)

_TAG_RE  = re.compile(r"(<[^>]*>)")            # HTML tags — never touched
_WORD_RE = re.compile(r"[A-Za-z" + _APOS + r"]+")   # a run to consider as one "word"


def _apply_case(latin: str, cyr: str) -> str:
    """Mirror the casing of `latin` onto the (1- or 2-char) `cyr` string."""
    if latin.isupper() and len(latin) > 1:
        return cyr.upper()
    if latin[:1].isupper():
        return cyr[0].upper() + cyr[1:]
    return cyr


def _replace(m: "re.Match") -> str:
    t = m.group(0)
    low = t.lower()

    # "Isʼhoq" -> "Исҳоқ": apostrophe here means "read s and h separately",
    # not the sh digraph. s and h are still individually transliterated.
    if len(t) == 3 and low[0] == "s" and low[2] == "h":
        return _apply_case(t[0], "с") + _apply_case(t[2], "ҳ")

    if len(t) == 2 and low[0] == "o" and t[1] in _APOS:
        return _apply_case(t, "ў")
    if len(t) == 2 and low[0] == "g" and t[1] in _APOS:
        return _apply_case(t, "ғ")

    if low == "ng":
        return _apply_case(t, "нг")

    if low in _DIGRAPH:
        return _apply_case(t, _DIGRAPH[low])

    if t in _APOS:
        return "ъ"  # tutuq belgisi elsewhere, e.g. sanʼat -> санъат

    if low == "e":
        prev = m.string[m.start() - 1] if m.start() > 0 else ""
        word_initial = not prev.isalpha()
        return _apply_case(t, "э" if word_initial else "е")

    if low in _SINGLE:
        return _apply_case(t, _SINGLE[low])

    return t  # bare c / w / anything else — leave untouched on purpose


def _word_is_recognized_uzbek(word: str) -> bool:
    """
    True only if EVERY character in `word` is consumed by a recognized
    Uzbek digraph/letter/mark — i.e. nothing in it would fall through to
    _replace()'s final "leave untouched" case.

    This gates transliteration at the WHOLE-WORD level rather than
    per-letter. Reason: a bare Latin "c" or "w" mid-word is a strong
    signal the "word" isn't really Uzbek at all (neither letter exists in
    the Uzbek Latin alphabet outside "ch") — most often a kept English
    abbreviation like "CSF" or "CT" sitting inside an otherwise-Uzbek
    card. Converting only the recognizable letters around it would leave
    a broken half-transliterated mess ("CSF" -> "CСФ"); leaving the whole
    word alone ("CSF" stays "CSF") is the safer failure mode.
    """
    pos = 0
    while pos < len(word):
        m = _TOKEN_RE.match(word, pos)
        if not m:
            return False
        t, low = m.group(0), m.group(0).lower()
        recognized = (
            low in _SINGLE
            or low in _DIGRAPH
            or low == "ng"
            or low == "e"
            or t in _APOS
            or (len(t) == 2 and t[1] in _APOS and low[0] in "og")
            or (len(t) == 3 and low[0] == "s" and low[2] == "h")
        )
        if not recognized:
            return False
        pos = m.end()
    return True


def _convert_word(word: str) -> str:
    return _TOKEN_RE.sub(_replace, word) if _word_is_recognized_uzbek(word) else word


def to_cyrillic(text: str) -> str:
    """Transliterate Latin-script Uzbek text to Cyrillic. HTML tags pass
    through untouched; text already in Cyrillic (or any other script) is
    left as-is; a Latin "word" containing a letter foreign to the Uzbek
    alphabet (bare c/w — almost always an abbreviation like CSF or CT) is
    left untouched as a whole rather than partially transliterated."""
    if not text:
        return text
    parts = _TAG_RE.split(text)
    return "".join(
        part if i % 2 else _WORD_RE.sub(lambda m: _convert_word(m.group(0)), part)
        for i, part in enumerate(parts)
    )


def has_cyrillic(text: str) -> bool:
    """True if `text` already contains any Cyrillic character."""
    return bool(text) and bool(re.search(r"[\u0400-\u04FF]", text))


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # (latin, expected_cyrillic)
        ("Barcha odamlar erkin, qadr-qimmat va huquqlarda teng boʻlib "
         "tugʻiladilar.",
         "Барча одамлар эркин, қадр-қиммат ва ҳуқуқларда тенг бўлиб "
         "туғиладилар."),
        ("Oʻzbekiston", "Ўзбекистон"),
        ("yangi", "янги"),
        ("yer", "ер"),
        ("yoq", "ёқ"),
        ("yulduz", "юлдуз"),
        ("sanʼat", "санъат"),
        ("maʼno", "маъно"),
        ("Isʼhoq", "Исҳоқ"),
        ("bosh miya", "бош мия"),
        ("yurak", "юрак"),
        ("KENGASH", "КЕНГАШ"),      # all-caps + ng digraph
        ("Gʻafur", "Ғафур"),         # capitalized gʻ digraph
        ("Bosh miyaning qaysi qismi CSF ishlab chiqaradi?",
         "Бош миянинг қайси қисми CSF ишлаб чиқаради?"),   # CSF kept whole, not "CСФ"
        ("CT va MRI natijalari", "CT ва МРИ натижалари"),   # CT kept (has bare C), rest converts
    ]

    failed = 0
    for latin, expected in cases:
        got = to_cyrillic(latin)
        ok = got == expected
        failed += not ok
        print(f"  {'✅' if ok else '❌'} {latin!r} -> {got!r}"
              + ("" if ok else f"  (expected {expected!r})"))

    # HTML / cloze safety
    html_in  = "The <b>oʻrta</b> layer<br>{{c1::yaʼni}} note"
    html_out = to_cyrillic(html_in)
    print(f"\n  HTML/cloze passthrough: {html_out!r}")
    assert "<b>" in html_out and "</b>" in html_out and "<br>" in html_out
    assert "{{c1::" in html_out and "}}" in html_out

    # Already-Cyrillic text should pass through unchanged
    already = "Бу матн аллақачон кирилл ёзувида."
    assert to_cyrillic(already) == already

    print(f"\n  {len(cases) - failed}/{len(cases)} word-level cases passed.")
    if failed:
        raise SystemExit(1)
