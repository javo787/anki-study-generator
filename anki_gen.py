"""
Anki Card Generator — YouTube Source
Requires: pip install google-genai youtube-transcript-api
"""

import re
import sys
import json
import time

from cards_json import parse_cards
from profiles import GERMAN_SYSTEM_PROMPT, build_explanation_instruction, german_card_cap

# ── IMPORTS ────────────────────────────────────────────────────────────────────

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    sys.exit("❌  Run: pip install youtube-transcript-api")

try:
    from google import genai
except ImportError:
    sys.exit("❌  Run: pip install google-genai")

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

MODEL      = "gemini-2.5-flash"
MAX_RETRY  = 3
RETRY_WAIT = 10  # seconds, multiplied by attempt number

# ── LANGUAGE ───────────────────────────────────────────────────────────────
# Shared by anki_gen.py, generator.py (fallback providers) and quality.py, so
# every prompt that writes or edits card text agrees on what "Card language"
# actually means. Previously `lang` was accepted by the pipeline's public
# functions but never reached any prompt — the picker in the UI had no
# effect at all on YouTube/PDF generation. This is what actually wires it up.
def build_language_instruction(lang: str) -> str:
    lang = (lang or "en").lower()
    if lang == "uz":
        return (
            'Write every "front" and "back" field in Uzbek, using the '
            "CYRILLIC alphabet (Ўзбек кирилл ёзуви) — NEVER the Latin/Roman "
            'alphabet. For example write "бош мия" and "юрак", never '
            '"bosh miya" or "yurak". Use standard Uzbek medical '
            "terminology; well-known international abbreviations (ДНК, МРТ, "
            "АТФ etc.) may stay in the form commonly used in Uzbek medical "
            "texts."
        )
    if lang == "ru":
        return 'Write every "front" and "back" field in Russian, using standard Russian medical terminology.'
    if lang == "de":
        return 'Write every "front" and "back" field in German (Deutsch).'
    return 'Write every "front" and "back" field in English.'


def build_profile_instruction(lang: str, profile: str = "medical") -> str:
    """
    The LANGUAGE rule for a prompt. For a medical deck "Card language" is the language of the
    whole card; for a German deck it is only the language of the translations (the German
    itself must stay German) — see profiles.build_explanation_instruction.
    """
    if profile == "german":
        return build_explanation_instruction(lang)
    return build_language_instruction(lang)


SYSTEM_PROMPT = """\
You are an expert medical Anki card creator trained on First Aid, Anki Mnemosyne decks, and Najeeb lectures.

CARD QUALITY RULES:
1. One card = one testable fact (anatomy, function, pathway, number, clinical correlation)
2. Front: specific question — What / Where / Which / How / Name / List
3. Back: clean bullet list (max 5 bullets) + 💡 mnemonic or key clinical note where genuinely useful
4. Avoid vague questions like "What is X?" — prefer "What are the branches of X?" or "Where does X drain?"
5. Skip filler, greetings, meta-commentary from the transcript
6. You are processing chunk {chunk_index} of {chunk_total} — do NOT repeat cards from previous chunks
7. Topic of this chunk: {topic}
8. Number of cards follows the content: one card per distinct testable fact, up to 35 per chunk.
   NEVER pad to reach a number — if the segment has few testable facts (intro, recap, housekeeping),
   return only those few cards.
9. FORMATTING: use HTML only, never markdown (no **bold**, use <b>bold</b>)
10. For "type": "cloze" cards, "front" MUST contain real Anki cloze syntax
    {{{{c1::answer}}}} — NEVER "___" placeholders. Anki only creates a card
    for each {{{{cN::...}}}} it finds in the text; without it, no card is made at all.
11. LANGUAGE: {language_instruction}
12. The text is an automatic speech-recognition transcript: it may lack punctuation and medical terms
    are often misspelled (e.g. "coroid plexus" = "choroid plexus"). Write every term with its correct
    standard spelling. Correct spelling only — never add or change facts the lecturer did not state.

OUTPUT: JSON only — no markdown fences, no explanation, no preamble.

{{
  "cards": [
    {{
      "front": "What are the three layers of the meninges from outside to inside?",
      "back": "• Dura mater\\n• Arachnoid mater\\n• Pia mater\\n💡 Mnemonic: <b>DAP</b> — Dura, Arachnoid, Pia",
      "type": "basic",
      "tags": ["neuroanatomy", "meninges"]
    }},
    {{
      "front": "What is the outermost layer of the meninges?<br><br>{{{{c1::Dura mater}}}}",
      "back": "💡 Dura = hard in Latin — it is the toughest layer",
      "type": "cloze",
      "tags": ["neuroanatomy", "meninges"]
    }}
  ]
}}"""

# ── TRANSCRIPT ─────────────────────────────────────────────────────────────────

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_YT_HOSTS = {"youtube.com", "youtube-nocookie.com"}


def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from a URL (or accept a bare 11-character ID).

    Handles watch?v=, youtu.be/, /embed/, /shorts/, /live/, /v/ and music./m.
    subdomains, with any extra query parameters. Anything that is not a YouTube
    video URL (other sites, playlists, channels) raises ValueError instead of
    guessing — the old regex search happily took "abv=XXXXXXXXXXX" or any
    site's "?v=" for a video id.
    """
    from urllib.parse import parse_qs, urlparse

    raw = (url or "").strip()
    if _VIDEO_ID.fullmatch(raw):
        return raw

    parsed = urlparse(raw if "://" in raw else "https://" + raw)
    host = (parsed.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    segments = [s for s in parsed.path.split("/") if s]

    candidate = None
    if host == "youtu.be":
        candidate = segments[0] if segments else None
    elif host in _YT_HOSTS or host.endswith(".youtube.com") or host.endswith(".youtube-nocookie.com"):
        if segments and segments[0] == "watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        elif len(segments) >= 2 and segments[0] in ("embed", "shorts", "live", "v", "e"):
            candidate = segments[1]

    if candidate and _VIDEO_ID.fullmatch(candidate):
        return candidate
    raise ValueError(f"Cannot extract video ID from: {url}")


def build_proxy_config(cfg: dict):
    """
    Build a youtube_transcript_api ProxyConfig from settings, or None.

    Why this exists: YouTube's (undocumented) transcript endpoint blocks
    most cloud-provider IPs outright — GitHub Codespaces, AWS, GCP, Azure —
    which is exactly what a RequestBlocked error means. It has nothing to
    do with language or video content; it happens before any of that is
    even looked at. The library's own maintainers recommend routing these
    specific requests through a proxy (ideally rotating residential IPs)
    rather than any code-side workaround, since there IS no code-side fix
    for "this IP is blocked".

    Webshare gets first-class support in youtube-transcript-api (retries
    with a fresh IP from the pool on a block), so it's tried first if
    configured. A single generic http(s) proxy URL — any provider, or a
    proxy you run yourself — is the fallback. Neither configured -> None,
    meaning requests go out from this machine's IP exactly as before.
    """
    from youtube_transcript_api.proxies import (
        GenericProxyConfig, WebshareProxyConfig, InvalidProxyConfig,
    )

    webshare_user = (cfg.get("webshare_proxy_username") or "").strip()
    webshare_pass = (cfg.get("webshare_proxy_password") or "").strip()
    generic_url   = (cfg.get("youtube_proxy_url") or "").strip()

    try:
        if webshare_user and webshare_pass:
            return WebshareProxyConfig(
                proxy_username=webshare_user, proxy_password=webshare_pass,
            )
        if generic_url:
            return GenericProxyConfig(http_url=generic_url, https_url=generic_url)
    except InvalidProxyConfig as e:
        print(f"  ⚠️  Proxy config ignored (invalid): {e}")
    return None


# YouTube captions carry non-speech noise that would otherwise end up inside the
# text the model reads: "[Music]", "[Applause]", "[ __ ]", ">>" speaker changes, ♪.
# Square brackets are only stripped when they hold plain words (no digits/symbols),
# so notation such as "[Ca2+]" is left alone.
_SOUND_TAG = re.compile(r"\[[A-Za-z _,'\-]{1,40}\]")
_MUSIC     = re.compile(r"[♪♫]+")
_SPEAKER   = re.compile(r">{2,}")


def _clean_caption(text: str) -> str:
    """One caption line -> clean spoken text ('' if nothing spoken is left)."""
    text = _SOUND_TAG.sub(" ", text or "")
    text = _MUSIC.sub(" ", text)
    text = _SPEAKER.sub(" ", text)
    return " ".join(text.split())


class NoCaptionsError(RuntimeError):
    """The video has no captions at all (turned off, or none exist).

    The one transcript failure that transcribing the AUDIO can fix — every other
    failure (blocked, private, age-restricted ...) would fail for the audio too."""


_CAUSE_STOP = re.compile(r"\s*(?:If you are sure|If this happens)", re.IGNORECASE)
_CAUSE_START = re.compile(r"This is most likely caused by:\s*", re.IGNORECASE)
LOCAL_VIDEO_TIP = "Or download the video and use the 'Local video' tab — Whisper transcribes straight from the audio."


def _short_cause(exc) -> str:
    """The one useful sentence of a youtube-transcript-api error (the library appends ~600
    characters of boilerplate, including a request to open an issue on GitHub)."""
    text = str(exc)
    m = _CAUSE_START.search(text)
    if m:
        text = text[m.end():]
    text = _CAUSE_STOP.split(text)[0]
    return " ".join(text.split())[:160]


def describe_transcript_error(exc, proxy_config=None) -> Exception:
    """Turn a youtube-transcript-api exception into a short error that says what is wrong
    and what to do. Only 'no captions' becomes NoCaptionsError (see above)."""
    try:
        from youtube_transcript_api import _errors as E
    except ImportError:                                           # pragma: no cover
        return RuntimeError(f"Could not fetch the transcript: {exc}")

    def is_a(*names):
        return isinstance(exc, tuple(getattr(E, n) for n in names if hasattr(E, n)))

    if is_a("TranscriptsDisabled", "NoTranscriptFound"):
        return NoCaptionsError("This video has no captions (the uploader turned them off or none exist).")
    if is_a("AgeRestricted"):
        return RuntimeError("This video is age-restricted: YouTube only gives captions to a logged-in account. "
                            "Download the video and use the 'Local video' tab.")
    if is_a("VideoUnplayable"):
        reason = getattr(exc, "reason", None) or _short_cause(exc)
        return RuntimeError(f"This video cannot be played: {reason}.")
    if is_a("VideoUnavailable"):
        return RuntimeError("This video is unavailable (deleted, private or blocked in your region).")
    if is_a("RequestBlocked", "IpBlocked"):
        if proxy_config is not None:
            return RuntimeError("YouTube blocked the request even through your proxy. Try another proxy "
                                "(residential proxies work best) or wait a while. " + LOCAL_VIDEO_TIP)
        return RuntimeError("YouTube is blocking transcript requests from this IP (common on cloud/VPN "
                            "connections). Add a proxy in Settings -> YouTube proxy. " + LOCAL_VIDEO_TIP)
    return RuntimeError(f"YouTube refused to give the captions ({_short_cause(exc) or exc.__class__.__name__}). "
                        "Try again later; a proxy (Settings -> YouTube proxy) or the 'Local video' tab are the workarounds.")


def get_transcript_entries(
    video_id: str,
    languages: list[str] | None = None,
    proxy_config=None,
) -> list[dict]:
    """
    Return raw transcript entries with timestamps.
    Each entry: {"text": str, "start": float, "duration": float}
    Used by chunker.py for smart splitting.

    languages: priority list of language codes to look for, e.g. ["uz", "en"].
    Defaults to ["en"] — matches the old hardcoded behaviour exactly, so any
    existing caller that doesn't pass `languages` is unaffected.

    proxy_config: optional youtube_transcript_api ProxyConfig (see
    build_proxy_config above) — routes requests through a proxy to work
    around YouTube blocking cloud-provider IPs (RequestBlocked/IpBlocked).
    None means "use this machine's IP directly", same as before.

    Root bug this fixes: this used to call fetcher.fetch(video_id) with no
    `languages` arg at all, which youtube-transcript-api silently defaults
    to ('en',). For any non-English video without an English caption track
    (i.e. most Russian/Uzbek/German lecture videos), that raised inside
    generate_from_youtube() regardless of what "Card language" was selected
    in the UI — the language picker never actually reached this call.

    Fetch strategy (each step only runs if the previous one fails):
      1. Fetch a transcript natively in one of `languages` (manually created
         captions are preferred over auto-generated ones by the library).
      2. No native transcript in any of `languages` — take whatever
         transcript IS available on the video and, if YouTube allows
         translating it, translate it into the first language requested.
      3. Translation isn't available either — fall back to whatever
         transcript exists, untranslated (still better than failing the
         whole pipeline; the generation step's own language instruction
         will still produce output in the requested language/script).
    """
    languages = list(languages) if languages else ["en"]
    fetcher   = YouTubeTranscriptApi(proxy_config=proxy_config)

    def _entries(fetched) -> list[dict]:
        out = []
        for e in fetched:
            text = _clean_caption(e.text)
            if text:
                out.append({"text": text, "start": e.start, "duration": e.duration})
        return out

    # ── 1. Direct match ────────────────────────────────────────────────────
    try:
        fetched = fetcher.fetch(video_id, languages=languages)
        print(f"  🌐  Transcript: native '{fetched.language_code}' track")
        return _entries(fetched)
    except Exception:
        pass

    # ── 2/3. Fall back to whatever exists, translating if possible ────────
    try:
        transcript_list = fetcher.list(video_id)
        available = transcript_list.find_transcript(
            [t.language_code for t in transcript_list]
        )
        wanted            = languages[0]
        translation_codes = {t.language_code for t in available.translation_languages}

        if available.language_code != wanted and available.is_translatable and wanted in translation_codes:
            fetched = available.translate(wanted).fetch()
            print(f"  🌐  Transcript: translated '{available.language_code}' → '{wanted}'")
        else:
            fetched = available.fetch()
            print(f"  🌐  Transcript: no '{'/'.join(languages)}' track — "
                  f"using '{available.language_code}' instead")
        return _entries(fetched)

    except Exception as e:
        raise describe_transcript_error(e, proxy_config) from e


def get_transcript(video_id: str, languages: list[str] | None = None) -> str:
    """Return plain text transcript. Backwards compatible."""
    entries = get_transcript_entries(video_id, languages=languages)
    return " ".join(e["text"] for e in entries)


# ── CARD GENERATION ────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> list[dict]:
    """Parse the model reply into cards (tolerant: see cards_json.py)."""
    return parse_cards(raw)


def generate_cards(
    chunk:       str,
    deck:        str,
    client,
    chunk_index: int = 1,
    chunk_total: int = 1,
    topic:       str = "general",
    lang:        str = "en",
    profile:     str = "medical",
) -> list[dict]:
    """
    Generate Anki cards from a text chunk using Gemini.

    Args:
        chunk:       text content to process
        deck:        deck name (used as context)
        client:      Gemini client instance
        chunk_index: current chunk number (for context)
        chunk_total: total chunks (for context)
        topic:       topic hint from pipeline (improves quality)
        lang:        output card language — "en" | "ru" | "de" | "uz"
                     ("uz" writes Cyrillic script, see build_language_instruction)
        profile:     "medical" (default) | "german" (textbook vocabulary & grammar,
                     see profiles.py) — selects the prompt

    Returns:
        list of card dicts with keys: front, back, type, tags
    """
    template = GERMAN_SYSTEM_PROMPT if profile == "german" else SYSTEM_PROMPT
    system = template.format(
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        topic=topic,
        language_instruction=build_profile_instruction(lang, profile),
        max_cards=german_card_cap(chunk),          # only the german templates use it
    )

    prompt = (
        f"Deck: {deck}\n"
        f"Chunk: {chunk_index} of {chunk_total}\n"
        f"Topic: {topic}\n\n"
        f"Transcript:\n{chunk}\n\n"
        f"Generate Anki cards. JSON only."
    )

    # заменённый одиночный вызов без внутреннего retry
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "system_instruction": system,
                "temperature": 0.3,
            },
        )
        return _parse_response(response.text)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error (chunk {chunk_index}): {e}")
        return []
    except Exception:
        raise  # пробрасываем исключение наверх в generate_chunk

    return []


# ── DEDUPLICATION ──────────────────────────────────────────────────────────────

def deduplicate(cards: list[dict]) -> list[dict]:
    """
    Remove exact duplicate cards by front text.
    Semantic deduplication is handled later by filter.py (Groq).
    """
    seen, unique = set(), []
    for c in cards:
        k = c["front"].lower().strip()
        if k not in seen:
            seen.add(k)
            unique.append(c)
    return unique


# ── CLIENT ─────────────────────────────────────────────────────────────────────

def build_client(api_key: str):
    """Build and return a Gemini client."""
    return genai.Client(api_key=api_key)