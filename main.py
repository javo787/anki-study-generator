"""
Anki Card Generator — Main Menu
Run: python main.py
"""

import os
import sys
from datetime import datetime

from config import load_config, save_config, mask_key

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "output"
VERSION     = "1.0"

# ── UI HELPERS ─────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    print(f"╔══════════════════════════════════════╗")
    print(f"║     🧠  Anki Card Generator  v{VERSION}     ║")
    print(f"╚══════════════════════════════════════╝")
    print()


def status_line(cfg: dict):
    key = "✅ set" if cfg["gemini_api_key"] else "⚠️  not set"
    print(f"  API key: {key}  │  Lang: {cfg['default_lang']}  │  Format: {cfg['default_format']}")
    print()


def ask(prompt: str, default: str = "") -> str:
    val = input(f"  {prompt}").strip()
    return val if val else default


def pause():
    input("\n  Press Enter to continue...")


def print_preview(cards: list):
    print("\n  ── Preview (first 3 cards) " + "─" * 18)
    for i, card in enumerate(cards[:3], 1):
        front = card["front"][:75] + ("…" if len(card["front"]) > 75 else "")
        back  = card["back"][:75].replace("\n", " │ ") + ("…" if len(card["back"]) > 75 else "")
        print(f"\n  [{i}] Q: {front}")
        print(f"      A: {back}")
    print("\n  " + "─" * 46)


def save_cards(cards: list, deck: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str  = datetime.now().strftime("%Y-%m-%d_%H-%M")
    safe_deck = "".join(c if c.isalnum() or c in "-_ " else "" for c in deck)
    safe_deck = safe_deck.strip().replace(" ", "_")
    filename  = os.path.join(OUTPUT_DIR, f"anki_{safe_deck}_{date_str}.txt")

    lines = [
        "#separator:tab",
        "#html:false",
        f"#deck:{deck}",
        "#notetype:Basic",
        "#columns:Front\tBack\tTags",
    ]
    for card in cards:
        front = card["front"].replace("\t", " ").replace("\n", "<br>")
        back  = card["back"].replace("\t", " ").replace("\n", "<br>")
        tags  = " ".join(card.get("tags", []))
        lines.append(f"{front}\t{back}\t{tags}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filename

# ── MENU: SETTINGS ─────────────────────────────────────────────────────────────

def menu_settings(cfg: dict):
    while True:
        clear()
        banner()
        print("  SETTINGS\n")
        print(f"  1.  Gemini API key   [{mask_key(cfg['gemini_api_key'])}]")
        print(f"  2.  Default language [{cfg['default_lang']}]")
        print(f"  3.  Default format   [{cfg['default_format']}]")
        print(f"  4.  Back\n")

        choice = ask("Choose: ")

        if choice == "1":
            key = ask("Paste Gemini API key: ")
            if key:
                cfg["gemini_api_key"] = key
                save_config(cfg)
                print("\n  ✅  Key saved.")
            pause()

        elif choice == "2":
            print("\n  Available: en / ru / de")
            lang = ask("Language: ").lower()
            if lang in ("en", "ru", "de"):
                cfg["default_lang"] = lang
                save_config(cfg)
                print("  ✅  Saved.")
            else:
                print("  ⚠️  Invalid choice.")
            pause()

        elif choice == "3":
            print("\n  Available: basic / cloze / both")
            fmt = ask("Format: ").lower()
            if fmt in ("basic", "cloze", "both"):
                cfg["default_format"] = fmt
                save_config(cfg)
                print("  ✅  Saved.")
            else:
                print("  ⚠️  Invalid choice.")
            pause()

        elif choice == "4":
            break

# ── MENU: YOUTUBE ──────────────────────────────────────────────────────────────

def menu_youtube(cfg: dict):
    clear()
    banner()
    print("  YOUTUBE → ANKI\n")

    if not cfg["gemini_api_key"]:
        print("  ⚠️  No API key set. Go to Settings first.")
        pause()
        return

    url = ask("YouTube URL: ")
    if not url:
        return

    deck = ask("Deck name: ", "Neuroanatomy")

    print(f"\n  Language [{cfg['default_lang']}] — Enter to keep or type new (en/ru/de): ", end="")
    lang_in = input().strip().lower()
    lang    = lang_in if lang_in in ("en", "ru", "de") else cfg["default_lang"]

    print("\n  ⏳  Fetching transcript...\n")

    try:
        from google import genai
        from anki_gen import (
            extract_video_id, get_transcript,
            chunk_text, generate_cards, deduplicate,
        )

        client   = genai.Client(api_key=cfg["gemini_api_key"])
        video_id = extract_video_id(url)

        transcript = get_transcript(video_id)
        print(f"  ✅  Transcript: {len(transcript):,} characters")

        chunks    = chunk_text(transcript, cfg["chunk_size"])
        all_cards = []

        for i, chunk in enumerate(chunks, 1):
            print(f"  🔄  Chunk {i}/{len(chunks)}...", end=" ", flush=True)
            cards = generate_cards(chunk, deck, client)
            print(f"{len(cards)} cards")
            all_cards.extend(cards)

        unique   = deduplicate(all_cards)
        filename = save_cards(unique, deck)

        print(f"\n  ✅  Total unique cards: {len(unique)}")
        print_preview(unique)
        print(f"\n  💾  Saved: {filename}")

    except Exception as e:
        print(f"\n  ❌  Error: {e}")

    pause()

# ── MENU: PDF ──────────────────────────────────────────────────────────────────

def menu_pdf(cfg: dict):
    clear()
    banner()
    print("  PDF → ANKI\n")

    if not cfg["gemini_api_key"]:
        print("  ⚠️  No API key set. Go to Settings first.")
        pause()
        return

    pdf_path = ask("PDF file path: ").strip('"')
    if not pdf_path or not os.path.exists(pdf_path):
        print("  ❌  File not found.")
        pause()
        return

    deck = ask("Deck name: ", "My Deck")

    all_pages = ask("All pages? (y/n): ").lower()
    if all_pages == "y":
        page_from, page_to = 1, 99999
    else:
        page_from = int(ask("From page: ") or "1")
        page_to   = int(ask("To page: ")   or "99999")

    print(f"\n  Language [{cfg['default_lang']}] — Enter to keep or type new (en/ru/de): ", end="")
    lang_in = input().strip().lower()
    lang    = lang_in if lang_in in ("en", "ru", "de") else cfg["default_lang"]

    print("\n  ⏳  Extracting PDF...\n")

    try:
        from google import genai
        from pdf_gen import (
            extract_pdf_text, chunk_text,
            generate_cards, detect_mode, deduplicate,
        )

        client = genai.Client(api_key=cfg["gemini_api_key"])
        mode   = detect_mode(deck)
        print(f"  🎯  Mode: {mode.upper()}")

        text, total = extract_pdf_text(pdf_path, page_from, page_to)
        print(f"  ✅  Extracted {len(text):,} characters from {total} pages")

        chunks    = chunk_text(text, cfg["chunk_size"])
        all_cards = []

        for i, chunk in enumerate(chunks, 1):
            print(f"  🔄  Chunk {i}/{len(chunks)}...", end=" ", flush=True)
            cards = generate_cards(chunk, mode, deck, lang, client)
            print(f"{len(cards)} cards")
            all_cards.extend(cards)

        unique   = deduplicate(all_cards)
        filename = save_cards(unique, deck)

        print(f"\n  ✅  Total unique cards: {len(unique)}")
        print_preview(unique)
        print(f"\n  💾  Saved: {filename}")

    except Exception as e:
        print(f"\n  ❌  Error: {e}")

    pause()

# ── MAIN LOOP ──────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    while True:
        clear()
        banner()
        status_line(cfg)

        print("  1.  YouTube video → Anki cards")
        print("  2.  PDF book → Anki cards")
        print("  3.  Settings")
        print("  4.  Exit")
        print()

        choice = ask("Choose (1-4): ")

        if choice == "1":
            menu_youtube(cfg)
        elif choice == "2":
            menu_pdf(cfg)
        elif choice == "3":
            menu_settings(cfg)
        elif choice == "4":
            print("\n  Goodbye!\n")
            sys.exit(0)

        cfg = load_config()


if __name__ == "__main__":
    main()