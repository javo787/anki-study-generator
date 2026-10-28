"""Offline fakes for the YouTube -> cards path. No network, no API keys."""
import json
import re
import types

from generator import AIClient


def block_entries(n_blocks=5, block_seconds=300, step=5.0, words_per_entry=10, tag="s"):
    """Caption-like transcript: n_blocks blocks of speech separated by a 3 s pause,
    so chunk_youtube() makes one chunk per block (block_seconds >= MIN_CHUNK_SECONDS)."""
    out, t, k = [], 0.0, 0
    for b in range(n_blocks):
        for _ in range(int(block_seconds / step)):
            k += 1
            words = " ".join(f"{tag}{k}w{j}" for j in range(words_per_entry - 4))
            out.append({"text": f"{words} the structure drains fluid.", "start": t, "duration": step})
            t += step
        t += 3.0
    return out


_SYL = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]          # 70 syllables


def pseudo(n: int) -> str:
    """A made-up 6-letter word, different (in its first 5 letters) for every n < 4900."""
    return _SYL[n % 70] + _SYL[(n // 70) % 70] + _SYL[(n // 4900) % 70]


def word_entries(n_entries=120, step=4.0, words_per_entry=5, t0=0.0):
    """Transcript whose every caption has its own distinctive words, so tests can say
    exactly which caption a card was 'taken from'."""
    return [{"text": " ".join(pseudo(k * words_per_entry + j) for j in range(words_per_entry)),
             "start": t0 + k * step, "duration": step} for k in range(n_entries)]


class Resp:
    def __init__(self, text):
        self.text = text


class FakeGemini:
    """Mimics google-genai: client.models.generate_content(model, contents, config).

    script(chunk_index, attempt_no) -> None (succeed) | Exception (raise) | str (raw reply)
    """
    def __init__(self, script=None):
        self.script = script or (lambda idx, attempt: None)
        self.attempts = {}
        self.models = self

    def generate_content(self, model, contents, config=None):
        if "main anatomical or medical topic" in contents:       # topic detection call
            return Resp("Fake topic")
        idx = int(re.search(r"Chunk: (\d+) of", contents).group(1))
        self.attempts[idx] = self.attempts.get(idx, 0) + 1
        outcome = self.script(idx, self.attempts[idx])
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            return Resp(outcome)
        text = contents.split("Transcript:\n", 1)[1].rsplit("\n\nGenerate Anki", 1)[0]
        cards = [{"front": f"What is {pseudo(idx * 10 + i)} in the {pseudo(idx * 10 + i + 500)} part?",
                  "back": f"{pseudo(idx * 10 + i + 1000)} {pseudo(idx * 10 + i + 2000)}", "type": "basic", "tags": []}
                 for i in range(3)]
        return Resp(json.dumps({"cards": cards}))


class FakeChatClient:
    """Mimics groq / openai clients: client.chat.completions.create(...)."""
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model, messages, temperature=0.3, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("500 fallback down")
        cards = [{"front": "Fallback card?", "back": "yes", "type": "basic", "tags": []}]
        msg = types.SimpleNamespace(content=json.dumps({"cards": cards}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def gemini_client(fake):
    return AIClient(name="Gemini (gemini-2.5-flash)", provider="gemini",
                    model="gemini-2.5-flash", client=fake, priority=1)


def groq_client(fake):
    return AIClient(name="Groq Llama", provider="groq", model="llama", client=fake, priority=3)


RATE_429 = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Quota exceeded for metric: "
            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 41.2s.', 'details': "
            "[{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '41s'}]}}")
DAILY_429 = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Quota exceeded for metric: "
             "GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 250. Please retry in 41.2s.', "
             "'details': [{'retryDelay': '41s'}]}}")


class Sleeper:
    """Stands in for time.sleep so tests are instant and record what was waited."""
    def __init__(self):
        self.calls = []

    def __call__(self, s):
        self.calls.append(s)
