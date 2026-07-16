"""
usage.py — API usage monitor and rate limit tracker
Tracks requests and tokens per provider, resets daily.
"""

import os
import json
import time
from datetime import datetime, date
from dataclasses import dataclass, field

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

USAGE_FILE  = "usage.json"

# Daily free tier limits
LIMITS = {
    "gemini_flash": {"requests": 1500, "tokens": 1_000_000},
    "gemini_pro":   {"requests": 50,   "tokens": 32_000},
    "groq":         {"requests": 14400,"tokens": 500_000},
    "openrouter":   {"requests": 200,  "tokens": 200_000},
    "cerebras":     {"requests": 14400,"tokens": 1_000_000},
}

PROVIDER_LABELS = {
    "gemini_flash": "Gemini Flash",
    "gemini_pro":   "Gemini Pro  ",
    "groq":         "Groq Llama  ",
    "openrouter":   "OpenRouter  ",
    "cerebras":     "Cerebras    ",
}

# ── DATA CLASSES ───────────────────────────────────────────────────────────────

@dataclass
class ProviderUsage:
    requests: int = 0
    tokens:   int = 0
    errors:   int = 0
    last_used: str = ""

    def to_dict(self) -> dict:
        return {
            "requests":  self.requests,
            "tokens":    self.tokens,
            "errors":    self.errors,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProviderUsage":
        return cls(
            requests  = d.get("requests",  0),
            tokens    = d.get("tokens",    0),
            errors    = d.get("errors",    0),
            last_used = d.get("last_used", ""),
        )


@dataclass
class UsageData:
    date:      str
    providers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "providers": {
                k: v.to_dict()
                for k, v in self.providers.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UsageData":
        providers = {
            k: ProviderUsage.from_dict(v)
            for k, v in d.get("providers", {}).items()
        }
        return cls(date=d.get("date", ""), providers=providers)


# ── CORE ───────────────────────────────────────────────────────────────────────

def _today() -> str:
    return date.today().isoformat()


def _load() -> UsageData:
    """Load usage data, reset if it's a new day."""
    if not os.path.exists(USAGE_FILE):
        return UsageData(date=_today())

    try:
        with open(USAGE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = UsageData.from_dict(raw)

        # New day — reset all counts
        if data.date != _today():
            print(f"  🔄  New day — resetting usage counters")
            return UsageData(date=_today())

        return data

    except Exception:
        return UsageData(date=_today())


def _save(data: UsageData):
    """Save usage data to file."""
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)


def _get_provider(data: UsageData, provider: str) -> ProviderUsage:
    """Get or create provider usage entry."""
    if provider not in data.providers:
        data.providers[provider] = ProviderUsage()
    return data.providers[provider]


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def record(provider: str, tokens: int = 0, error: bool = False):
    """
    Record one API call for a provider.

    Args:
        provider: "gemini_flash" | "gemini_pro" | "groq" | "openrouter" | "cerebras"
        tokens:   estimated tokens used (input + output)
        error:    whether the call resulted in an error
    """
    data = _load()
    p    = _get_provider(data, provider)

    p.requests  += 1
    p.tokens    += tokens
    p.last_used  = datetime.now().strftime("%H:%M:%S")
    if error:
        p.errors += 1

    _save(data)


def get_remaining(provider: str) -> dict:
    """
    Get remaining requests and tokens for a provider today.

    Returns:
        {"requests": int, "tokens": int, "requests_pct": float, "tokens_pct": float}
    """
    data   = _load()
    p      = _get_provider(data, provider)
    limits = LIMITS.get(provider, {"requests": 999, "tokens": 999999})

    req_remaining = max(0, limits["requests"] - p.requests)
    tok_remaining = max(0, limits["tokens"]   - p.tokens)
    req_pct       = req_remaining / limits["requests"] * 100
    tok_pct       = tok_remaining / limits["tokens"]   * 100

    return {
        "requests":     req_remaining,
        "tokens":       tok_remaining,
        "requests_pct": round(req_pct, 1),
        "tokens_pct":   round(tok_pct, 1),
    }


def is_available(provider: str, needed_tokens: int = 1000) -> bool:
    """
    Check if a provider has enough quota for another request.
    Used by pipeline.py to decide which AI to use.
    """
    remaining = get_remaining(provider)
    return (
        remaining["requests"] > 0 and
        remaining["tokens"]   >= needed_tokens
    )


def get_best_available(providers: list[str], needed_tokens: int = 1000) -> str | None:
    """
    Return the first available provider from the list (priority order).
    Returns None if all are exhausted.
    """
    for provider in providers:
        if is_available(provider, needed_tokens):
            return provider
    return None


def get_all_usage() -> dict:
    """
    Return full usage summary for all providers.
    Used by server.py for web UI display.
    """
    data   = _load()
    result = {}

    for provider, limits in LIMITS.items():
        p   = _get_provider(data, provider)
        req_used = p.requests
        tok_used = p.tokens
        req_lim  = limits["requests"]
        tok_lim  = limits["tokens"]

        result[provider] = {
            "label":          PROVIDER_LABELS.get(provider, provider),
            "requests_used":  req_used,
            "requests_limit": req_lim,
            "requests_pct":   round(req_used / req_lim * 100, 1),
            "tokens_used":    tok_used,
            "tokens_limit":   tok_lim,
            "tokens_pct":     round(tok_used / tok_lim * 100, 1),
            "errors":         p.errors,
            "last_used":      p.last_used,
            "available":      is_available(provider),
        }

    return result


def print_usage():
    """Print usage table to terminal."""
    usage = get_all_usage()

    print("\n  ┌──────────────────┬───────────────────┬───────────────────┐")
    print(  "  │ Provider         │ Requests          │ Tokens            │")
    print(  "  ├──────────────────┼───────────────────┼───────────────────┤")

    for provider, data in usage.items():
        req_bar = _bar(data["requests_pct"])
        tok_bar = _bar(data["tokens_pct"])
        status  = "✅" if data["available"] else "❌"

        print(
            f"  │ {status} {data['label']} │ "
            f"{data['requests_used']:>5}/{data['requests_limit']:<6} {req_bar} │ "
            f"{_fmt_tokens(data['tokens_used']):>6}/{_fmt_tokens(data['tokens_limit']):<6} {tok_bar} │"
        )

    print("  └──────────────────┴───────────────────┴───────────────────┘")
    print(f"  Last updated: {datetime.now().strftime('%H:%M:%S')} | Resets at midnight\n")


def reset_usage(provider: str = None):
    """
    Reset usage counters.
    If provider is None — reset all.
    """
    data = _load()

    if provider:
        if provider in data.providers:
            data.providers[provider] = ProviderUsage()
            print(f"  ✅  Reset usage for {provider}")
    else:
        data.providers = {}
        print("  ✅  Reset all usage counters")

    _save(data)


# ── HELPERS ────────────────────────────────────────────────────────────────────

def _bar(pct_remaining: float, width: int = 6) -> str:
    """Generate ASCII progress bar from remaining percentage."""
    filled = round(pct_remaining / 100 * width)
    empty  = width - filled
    return f"{'█' * filled}{'░' * empty}"


def _fmt_tokens(n: int) -> str:
    """Format token count as human-readable string."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


# ── DEBUG ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        provider = sys.argv[2] if len(sys.argv) > 2 else None
        reset_usage(provider)
    else:
        # Simulate some usage for demo
        record("gemini_flash", tokens=1200)
        record("gemini_flash", tokens=980)
        record("groq",         tokens=3400)
        record("gemini_pro",   tokens=8000)
        record("openrouter",   tokens=0, error=True)

        print_usage()