"""
profiles.py — what KIND of material is being turned into cards.

  "medical"  lectures / textbooks for medical students (the default; its prompts live in
             anki_gen.py, generator.py, filter.py and quality.py and are unchanged)
  "german"   a German-as-a-foreign-language textbook such as "Menschen" A1-A2:
             vocabulary (article + plural, Perfekt) and grammar patterns

Why a profile and not just "a different prompt": the medical prompts are wrong for a
language deck at every step, not only at generation —

  * generation   asks for medical facts
  * filter       calls vocabulary "trivial for a medical student" and drops it
  * quality      "fixes medical inaccuracies" and, with Card language = Uzbek, would
                 rewrite the German front of every card into Uzbek
  * format pass  would transliterate the German words themselves into Cyrillic
  * topic        asks for an anatomical topic

So the profile is threaded through the whole pipeline. This module is pure data and has
no imports from the rest of the project (so every other module can import it).

The German prompts descend from LANGUAGE_PROMPT in pdf_gen.py, which was only reachable
from the old terminal menu. Changes: real Anki cloze syntax instead of "___ (answer)",
the translation language follows "Card language" instead of a hard-coded "English +
Russian", no forced minimum number of cards, and an explicit "never guess a plural".
"""

DEFAULT_PROFILE = "medical"
PROFILES = ("medical", "german")


def normalize_profile(value) -> str:
    """None / "" -> the default; unknown values raise ValueError (fail early, not mid-run)."""
    if value is None or value == "":
        return DEFAULT_PROFILE
    name = str(value).strip().lower() if isinstance(value, str) else None
    if name in PROFILES:
        return name
    raise ValueError(f"Unknown profile {value!r}. Choose one of: {', '.join(PROFILES)}.")


# ── what "Card language" means for a language deck ──────────────────────────────

_KEEP_GERMAN = ("The German words, phrases and example sentences themselves always stay in German "
                "(Latin script) — only the translation and the explanation are written in the "
                "language named here. ")


def build_explanation_instruction(lang: str) -> str:
    """
    For the medical profile "Card language" is the language of the whole card. For a German
    deck it is the language of the TRANSLATIONS — the German on the front must stay German.
    """
    lang = (lang or "en").lower()
    if lang == "uz":
        return (_KEEP_GERMAN + "Write translations and explanations in Uzbek using the CYRILLIC "
                "alphabet (Ўзбек кирилл ёзуви) — never the Latin alphabet. For example write "
                '"шифокор", never "shifokor".')
    if lang == "ru":
        return _KEEP_GERMAN + "Write translations and explanations in Russian."
    if lang == "de":
        return ("Do not translate. Explain each word with a short, simple German definition or a "
                "synonym (A1-A2 vocabulary only). Everything on the card is in German.")
    return _KEEP_GERMAN + "Write translations and explanations in English."


# ── generation (Gemini) ─────────────────────────────────────────────────────────
# Filled with str.format(chunk_index, chunk_total, topic, language_instruction):
# literal braces are doubled, and the cloze example needs FOUR so the model sees {{c1::...}}.

GERMAN_SYSTEM_PROMPT = """\
You are an expert German-as-a-foreign-language Anki card creator. The learner studies from a \
textbook (for example "Menschen" A1-A2); the text below was extracted from some of its pages: \
vocabulary lists, dialogues, grammar boxes and exercises.

WHAT TO TURN INTO CARDS
1. Nouns — always WITH the article and the plural: "der Arzt, die Ärzte". For professions with \
a female form give both: "der Arzt / die Ärztin".
2. Verbs — the infinitive; add the Perfekt (haben/sein + Partizip II) whenever you are sure of \
it; mark separable verbs (an|kommen).
3. Adjectives, adverbs, prepositions, question words — the word itself.
4. Useful phrases and collocations — the complete phrase ("Wie geht's?").
5. Grammar patterns from grammar boxes (word order, verb endings, articles, cases) — as cloze cards.

WHAT TO IGNORE
Exercise instructions ("Ergänzen Sie ...", "Hören Sie ...", "Kreuzen Sie an ..."), empty answer \
lines, page and track numbers, headings, figure captions, and the names of people in dialogues.

CARD TYPES
- Vocabulary card ("type": "basic"): "front" = the German word or phrase (nouns with article); \
"back" = the translation, then for nouns "Plural: die ...", then ONE short example sentence in \
German, preferably taken from the text. Separate lines with <br>.
- Cloze card ("type": "cloze"): take an example sentence from the text and hide the key word or \
ending, using real Anki syntax: "Ich {{{{c1::bin}}}} ins Kino gegangen." Never use underscore \
(___) placeholders — Anki makes no card from them.

RULES
1. One card = one word, phrase or rule.
2. NEVER guess. If you are not sure of an article, a plural or a Perfekt form, leave that part \
out — a missing detail is better than a wrong one that gets memorised.
3. Example sentences must be simple (A1-A2) and must not contradict the textbook.
4. The number of cards follows the content: one card per item worth learning, up to 40 per \
chunk. NEVER pad to reach a number — if the pages hold little new material, return few cards.
5. You are processing chunk {chunk_index} of {chunk_total}; its topic is: {topic}
6. FORMATTING: HTML only, never markdown (no **bold** — use <b>bold</b>; <br> for line breaks).
7. LANGUAGE: {language_instruction}
8. The text comes from a PDF: line breaks, hyphenation and column order may be broken. Repair \
obvious extraction errors (for example "Ar zt" -> "Arzt").
9. Tags: short lowercase English words, e.g. vocabulary, nouns, verbs, phrase, grammar.

OUTPUT: JSON only — no markdown fences, no explanation.
{{
  "cards": [
    {{
      "front": "der Arzt / die Ärztin",
      "back": "<translation><br>Plural: die Ärzte<br>Der Arzt untersucht den Patienten.",
      "type": "basic",
      "tags": ["vocabulary", "nouns"]
    }},
    {{
      "front": "Ich {{{{c1::bin}}}} gestern ins Kino gegangen.",
      "back": "<translation of the sentence><br>Perfekt of «gehen»: ist gegangen",
      "type": "cloze",
      "tags": ["grammar", "perfekt"]
    }}
  ]
}}"""


# ── generation (Groq / NVIDIA / OpenRouter fallbacks) — same rules, shorter ─────

GERMAN_FALLBACK_PROMPT = """\
You are an expert German-as-a-foreign-language Anki card creator. The text comes from a \
textbook (for example "Menschen" A1-A2).

RULES:
1. One card = one word, phrase or rule
2. Nouns WITH article and plural ("der Arzt, die Ärzte"); verbs as infinitive + Perfekt when you \
are sure; useful phrases complete; grammar patterns as cloze cards
3. Vocabulary card ("type": "basic"): front = the German word/phrase, back = translation, \
"Plural: die ...", one short German example sentence. Separate lines with <br>
4. Cloze card ("type": "cloze"): hide the key word of an example sentence using real Anki \
syntax, for example "Ich {{{{c1::bin}}}} ins Kino gegangen." — never underscore (___) placeholders
5. NEVER guess an article, plural or Perfekt form — leave it out if you are not sure
6. Ignore exercise instructions ("Ergänzen Sie ...", "Hören Sie ..."), page/track numbers, \
headings and the names of people in dialogues
7. The number of cards follows the content, up to 40. NEVER pad to reach a number
8. Topic: {topic} | Chunk {chunk_index} of {chunk_total}
9. LANGUAGE: {language_instruction}
10. FORMATTING: HTML only, never markdown (no **bold** — use <b>bold</b>, and <br> for line breaks)
11. The text comes from a PDF and may have broken line breaks and hyphenation — repair obvious \
extraction errors

OUTPUT: JSON only — no markdown, no explanation.
{{"cards": [{{"front": "...", "back": "...", "type": "basic", "tags": ["vocabulary"]}}]}}"""


# ── topic (plain string, not formatted) ─────────────────────────────────────────

GERMAN_TOPIC_PROMPT = """\
In 3-6 words, name the main topic of this German textbook text: a vocabulary theme or a grammar point.
Examples: "Berufe und Arbeitsplatz", "Perfekt mit haben und sein", "Im Restaurant bestellen"
Reply with the topic only — no punctuation, no explanation."""


# ── duplicates (plain string, not formatted) — same JSON schema as filter.DEDUP_SYSTEM ──

GERMAN_DEDUP_SYSTEM = """\
You are reviewing German-learning Anki flashcards for duplicates.

You get several GROUPS of cards that look alike. Each card has an integer id.
Inside a group, two cards are duplicates only if they teach the SAME word, phrase or rule AND are \
the same kind of card (vocabulary card and vocabulary card, or two cloze cards of the same sentence).

NOT duplicates — keep both:
  - a vocabulary card and a cloze card made from its example sentence
  - two different words that happen to share an example sentence
  - the same word in two different meanings

For each group keep the BETTER card (complete article / plural / Perfekt, clearer example) and \
list the ids to remove. Use ONLY the ids you were given. Never write card text.

OUTPUT: JSON only — no markdown, no explanation. Use an empty "remove" list when a group has no \
duplicates.
{"results": [{"group": 0, "keep": 12, "remove": [40]}]}"""


# ── quality (Gemini Pro / Flash) — filled with str.format(language_instruction) ─

GERMAN_QUALITY_SYSTEM = """\
You are an experienced teacher of German as a foreign language (CEFR A1-B1) checking Anki \
flashcards for correctness.

Your tasks for EACH card:
1. ARTICLE & PLURAL — check der/die/das and the plural form of every noun; fix mistakes
2. VERB FORMS — check the infinitive, the Perfekt auxiliary (haben/sein) and the Partizip II; \
fix mistakes
3. EXAMPLE SENTENCES — they must be correct, natural German at A1-A2 level; fix grammar and \
word-order mistakes
4. TRANSLATION — it must be correct. LANGUAGE — {language_instruction}
5. CLOZE FORMAT — for "type": "cloze" cards, "front" MUST use real Anki syntax: \
"Ich {{{{c1::bin}}}} ins Kino gegangen." NEVER "___" placeholders — Anki makes no card from them
6. FORMATTING — HTML only, never markdown (use <b>bold</b> and <br> for line breaks)

IMPORTANT:
- If you are not sure that something is wrong, leave it exactly as it is. Change only what is \
certainly wrong.
- The German word or phrase on the front stays German — never translate the front into another \
language
- Do NOT remove cards
- Do NOT change card type
- Do NOT add mnemonics
- Preserve all existing tags
- Set "improved": true only for cards you actually changed

OUTPUT: JSON only — no markdown.
{{
  "cards": [
    {{
      "front": "...",
      "back": "...",
      "type": "basic",
      "tags": ["..."],
      "improved": true,
      "mnemonic_added": false
    }}
  ]
}}"""
