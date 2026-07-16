"""
chunker.py — Smart transcript and PDF chunker
Splits content at natural boundaries, not arbitrary character counts.
"""

import os
import re
from dataclasses import dataclass, field

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

# YouTube chunking
PAUSE_THRESHOLD    = 2.5   # seconds — gap between entries = topic boundary
MAX_CHUNK_SECONDS  = 600   # 10 minutes max per chunk
MIN_CHUNK_SECONDS  = 120   # 2 minutes min per chunk

# PDF chunking
PDF_MIN_CHUNK_CHARS = 1500  # minimum chars per chunk
PDF_MAX_CHUNK_CHARS = 6000  # maximum chars per chunk

# Heading patterns for PDF section detection
HEADING_PATTERNS = [
    r"^(Kapitel|Chapter|Unit|Lektion|Leçon|Урок)\s+\d+",  # Kapitel 1, Chapter 3
    r"^\d+\.\d*\s+[A-ZА-ЯÄÖÜa-z]",                        # 1.1 Introduction
    r"^[A-ZÄÖÜ][A-ZÄÖÜ\s]{4,}$",                           # ALL CAPS HEADING
    r"^\d+\s+[A-ZА-ЯÄÖÜ]",                                 # 1 Introduction
]

# ── DATA CLASSES ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text:       str
    index:      int          # chunk number (1-based)
    total:      int          # total chunks in this source
    source:     str          # "youtube" or "pdf"
    start_time: float = 0.0  # YouTube: start time in seconds
    end_time:   float = 0.0  # YouTube: end time in seconds
    page_from:  int   = 0    # PDF: first page
    page_to:    int   = 0    # PDF: last page
    topic_hint: str   = ""   # filled later by pipeline (Gemini)

    @property
    def duration_min(self) -> float:
        return (self.end_time - self.start_time) / 60

    @property
    def time_label(self) -> str:
        if self.source != "youtube":
            return ""
        s = int(self.start_time)
        e = int(self.end_time)
        return f"{s//60:02d}:{s%60:02d} – {e//60:02d}:{e%60:02d}"

    def summary(self) -> str:
        if self.source == "youtube":
            return (
                f"Chunk {self.index}/{self.total} | "
                f"{self.time_label} | "
                f"{self.duration_min:.1f} min | "
                f"{len(self.text):,} chars"
            )
        return (
            f"Chunk {self.index}/{self.total} | "
            f"Pages {self.page_from}–{self.page_to} | "
            f"{len(self.text):,} chars"
        )


# ── YOUTUBE CHUNKER ────────────────────────────────────────────────────────────

def chunk_youtube(entries: list[dict]) -> list[Chunk]:
    """
    Split YouTube transcript entries into smart chunks.

    entries: list of {"text": str, "start": float, "duration": float}

    Strategy:
      - Gap between entries > PAUSE_THRESHOLD → potential boundary
      - Never exceed MAX_CHUNK_SECONDS
      - Never split below MIN_CHUNK_SECONDS (merge small chunks forward)
    """
    if not entries:
        raise ValueError("Transcript is empty.")

    # ── Find all natural boundaries ────────────────────────────────────────────
    boundaries = []   # indices where a new chunk should start
    boundaries.append(0)

    for i in range(1, len(entries)):
        prev_end = entries[i - 1]["start"] + entries[i - 1].get("duration", 0)
        curr_start = entries[i]["start"]
        gap = curr_start - prev_end

        chunk_start_time = entries[boundaries[-1]]["start"]
        current_duration = curr_start - chunk_start_time

        # Force split if max duration exceeded
        if current_duration >= MAX_CHUNK_SECONDS:
            boundaries.append(i)
            continue

        # Natural pause boundary (only if we're past the minimum)
        if gap >= PAUSE_THRESHOLD and current_duration >= MIN_CHUNK_SECONDS:
            boundaries.append(i)

    # ── Build raw chunks from boundaries ──────────────────────────────────────
    raw_chunks = []
    boundaries.append(len(entries))  # sentinel

    for b in range(len(boundaries) - 1):
        start_idx = boundaries[b]
        end_idx   = boundaries[b + 1]
        segment   = entries[start_idx:end_idx]

        text       = " ".join(e["text"].strip() for e in segment)
        start_time = segment[0]["start"]
        last_entry = segment[-1]
        end_time   = last_entry["start"] + last_entry.get("duration", 0)

        raw_chunks.append({
            "text":       text,
            "start_time": start_time,
            "end_time":   end_time,
        })

    # ── Merge chunks that are too short ───────────────────────────────────────
    merged = []
    buffer = None

    for rc in raw_chunks:
        duration = rc["end_time"] - rc["start_time"]

        if buffer is None:
            buffer = rc.copy()
        elif (buffer["end_time"] - buffer["start_time"]) < MIN_CHUNK_SECONDS:
            # Merge into buffer
            buffer["text"]     += " " + rc["text"]
            buffer["end_time"]  = rc["end_time"]
        else:
            merged.append(buffer)
            buffer = rc.copy()

    if buffer:
        merged.append(buffer)

    # ── Wrap in Chunk dataclass ────────────────────────────────────────────────
    total = len(merged)
    result = []
    for i, m in enumerate(merged, 1):
        result.append(Chunk(
            text       = m["text"],
            index      = i,
            total      = total,
            source     = "youtube",
            start_time = m["start_time"],
            end_time   = m["end_time"],
        ))

    return result


# ── PDF CHUNKER ────────────────────────────────────────────────────────────────

def _is_heading(line: str) -> bool:
    """Check if a line looks like a section heading."""
    line = line.strip()
    if not line or len(line) > 120:
        return False
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, line):
            return True
    return False


def _extract_pages(path: str, page_from: int, page_to: int) -> list[dict]:
    """
    Extract pages from PDF as list of {"page": int, "text": str}.
    Requires pdfplumber.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Run: pip install pdfplumber")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        p_from = max(1, page_from)
        p_to   = min(total, page_to)

        for i in range(p_from - 1, p_to):
            text = pdf.pages[i].extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})

    if not pages:
        raise RuntimeError(
            "No text extracted. PDF may be scanned. Try OCR first."
        )

    return pages


def chunk_pdf(
    path: str,
    page_from: int = 1,
    page_to: int = 99999,
) -> list[Chunk]:
    """
    Split PDF into smart chunks using heading detection.

    Strategy:
      1. Try to split at headings (Kapitel, Chapter, Unit, etc.)
      2. If no headings found — split at paragraph boundaries
      3. Enforce min/max char limits by merging or splitting
    """
    pages = _extract_pages(path, page_from, page_to)
    total_pages = len(pages)

    # ── Try heading-based splitting ────────────────────────────────────────────
    sections = []     # list of {"pages": [int], "text": str}
    current_pages = []
    current_lines = []
    found_headings = False

    for page_data in pages:
        lines = page_data["text"].split("\n")
        for line in lines:
            if _is_heading(line) and current_lines:
                # Save current section
                sections.append({
                    "pages": current_pages.copy(),
                    "text":  "\n".join(current_lines).strip(),
                })
                current_pages = [page_data["page"]]
                current_lines = [line]
                found_headings = True
            else:
                if page_data["page"] not in current_pages:
                    current_pages.append(page_data["page"])
                current_lines.append(line)

    # Flush last section
    if current_lines:
        sections.append({
            "pages": current_pages,
            "text":  "\n".join(current_lines).strip(),
        })

    # ── Fallback: paragraph-based splitting ───────────────────────────────────
    if not found_headings:
        sections = []
        for page_data in pages:
            paragraphs = re.split(r"\n{2,}", page_data["text"])
            for para in paragraphs:
                para = para.strip()
                if para:
                    sections.append({
                        "pages": [page_data["page"]],
                        "text":  para,
                    })

    # ── Merge sections to respect min/max char limits ─────────────────────────
    merged = []
    buffer_text  = ""
    buffer_pages = []

    for sec in sections:
        combined_len = len(buffer_text) + len(sec["text"])

        if not buffer_text:
            buffer_text  = sec["text"]
            buffer_pages = sec["pages"].copy()

        elif combined_len <= PDF_MAX_CHUNK_CHARS:
            buffer_text  += "\n\n" + sec["text"]
            buffer_pages += [p for p in sec["pages"] if p not in buffer_pages]

        else:
            if len(buffer_text) >= PDF_MIN_CHUNK_CHARS:
                merged.append({
                    "text":  buffer_text,
                    "pages": sorted(buffer_pages),
                })
                buffer_text  = sec["text"]
                buffer_pages = sec["pages"].copy()
            else:
                # Buffer too small — merge anyway
                buffer_text  += "\n\n" + sec["text"]
                buffer_pages += [p for p in sec["pages"] if p not in buffer_pages]

    # Flush last buffer
    if buffer_text.strip():
        merged.append({
            "text":  buffer_text,
            "pages": sorted(buffer_pages),
        })

    # ── Wrap in Chunk dataclass ────────────────────────────────────────────────
    total = len(merged)
    result = []
    for i, m in enumerate(merged, 1):
        pages_list = m["pages"]
        result.append(Chunk(
            text      = m["text"],
            index     = i,
            total     = total,
            source    = "pdf",
            page_from = pages_list[0]  if pages_list else page_from,
            page_to   = pages_list[-1] if pages_list else page_to,
        ))

    return result


# ── DEBUG ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python chunker.py youtube <video_id>")
        print("  python chunker.py pdf <path> [from_page] [to_page]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "youtube":
        from anki_gen import get_transcript_entries
        video_id = sys.argv[2]
        print(f"Fetching transcript for: {video_id}")
        entries = get_transcript_entries(video_id)
        chunks  = chunk_youtube(entries)
        print(f"\nTotal chunks: {len(chunks)}\n")
        for c in chunks:
            print(f"  {c.summary()}")
            print(f"  Preview: {c.text[:80]}...\n")

    elif mode == "pdf":
        path      = sys.argv[2]
        page_from = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        page_to   = int(sys.argv[4]) if len(sys.argv) > 4 else 99999
        print(f"Chunking PDF: {path} pages {page_from}–{page_to}")
        chunks = chunk_pdf(path, page_from, page_to)
        print(f"\nTotal chunks: {len(chunks)}\n")
        for c in chunks:
            print(f"  {c.summary()}")
            print(f"  Preview: {c.text[:80]}...\n")