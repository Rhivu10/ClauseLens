from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import re

import fitz  # PyMuPDF


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Line:
    text: str
    page: int
    block: int
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_name: str
    bold: bool
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class HeadingInfo:
    level: int          # 0 = top, 1 = second, etc.
    text: str
    is_numbered: bool   # True if detected by numbering pattern


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    heading_path: List[str]  # e.g., ["Policy Statement", "Scope"]


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_lines(pdf_path: str) -> List[Line]:
    """
    Extract every text line from the PDF with layout metadata.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {pdf_path.suffix}")

    lines: List[Line] = []
    doc = fitz.open(pdf_path)
    try:
        for page_number, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            for block_number, block in enumerate(page_dict.get("blocks", [])):
                if block.get("type") != 0:  # skip images
                    continue
                for line in block.get("lines", []):
                    parts = []
                    sizes = []
                    fonts = []
                    bold_flags = []

                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text.strip():
                            continue
                        parts.append(text)
                        sizes.append(span.get("size", 0.0))
                        fonts.append(span.get("font", ""))
                        font_lower = span.get("font", "").lower()
                        bold_flags.append(
                            "bold" in font_lower
                            or "black" in font_lower
                            or "heavy" in font_lower
                        )

                    if not parts:
                        continue

                    text = "".join(parts).strip()
                    if not text:
                        continue

                    bbox = tuple(line.get("bbox", (0, 0, 0, 0)))
                    x0, y0, x1, y1 = bbox

                    lines.append(
                        Line(
                            text=text,
                            page=page_number,
                            block=block_number,
                            bbox=bbox,
                            font_size=max(sizes) if sizes else 0.0,
                            font_name=fonts[0] if fonts else "",
                            bold=any(bold_flags),
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                        )
                    )
    finally:
        doc.close()
    return lines


# ============================================================
# HEADING DETECTION
# ============================================================

class HeadingDetector:
    """
    Detects headings using two complementary signals:
      1. Numbering patterns (lightweight regex, extensible)
      2. Visual formatting (font size, boldness, line length)
    """

    def __init__(self, lines: List[Line]):
        self.lines = lines
        self.body_font_size = self._estimate_body_font_size()

        # Regex patterns for numbered headings.
        # Each pattern returns (level, heading_text) from a match.
        # Level 0 = top (e.g., "Policy 1.0", "Article I"), 1 = second, etc.
        self.numbering_patterns = [
            # 1.0, 2.0, 3.1 (many policies use decimal numbering)
            (re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$"), 1),
            # 1. 2. 3. (simple numbered)
            (re.compile(r"^\s*(\d+)\.\s+(.+)$"), 1),
            # 1.1, 1.2.3 (subsection without trailing dot)
            (re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.+)$"), 2),
            # A. B. C. (uppercase letter)
            (re.compile(r"^\s*([A-Z])\.\s+(.+)$"), 1),
            # (a) (b) (c)
            (re.compile(r"^\s*\(([a-z])\)\s+(.+)$", re.IGNORECASE), 3),
            # a) b) c)
            (re.compile(r"^\s*([a-z])\)\s+(.+)$", re.IGNORECASE), 3),
            # Article I, Chapter 2, Section 1.2
            (re.compile(r"^\s*(ARTICLE|CHAPTER|SECTION)\s+([A-Z0-9]+(?:\.[0-9]+)?)\s*[.:\-]?\s*(.*)$", re.IGNORECASE), 0),
        ]

    def _estimate_body_font_size(self) -> float:
        sizes = [
            round(line.font_size, 1)
            for line in self.lines
            if line.font_size > 0
        ]
        if not sizes:
            return 0.0
        return Counter(sizes).most_common(1)[0][0]

    def _is_visual_heading(self, line: Line) -> bool:
        """A line that looks like a heading but has no numbering."""
        text = line.text.strip()
        if not text:
            return False
        if len(text) > 120:          # headings are usually short
            return False

        # Significantly larger than body text
        if line.font_size >= self.body_font_size * 1.3:
            return True

        # Bold and short
        if line.bold and len(text.split()) <= 12:
            return True

        return False

    def _visual_level(self, line: Line) -> int:
        """Assign a relative level to a visual heading based on font size."""
        ratio = line.font_size / self.body_font_size if self.body_font_size else 1.0
        if ratio >= 1.5:
            return 0
        if ratio >= 1.2:
            return 1
        return 2   # bold short lines often act as lower-level headings

    def detect(self, line: Line) -> Optional[HeadingInfo]:
        """
        Return a HeadingInfo if the line is a heading, else None.
        """
        text = line.text.strip()
        if not text:
            return None

        # 1. Try numbering patterns first (strongest signal)
        for pattern, default_level in self.numbering_patterns:
            match = pattern.match(text)
            if match:
                # Determine level from the pattern
                # For patterns with explicit level, use default_level
                # For decimal patterns, count dots to get level
                groups = match.groups()
                if default_level == 1 and groups and "." in groups[0]:
                    # e.g., "1.1" -> level 2, "1.1.1" -> level 3
                    level = groups[0].count(".") + 1
                else:
                    level = default_level
                # The full heading text is the whole line
                return HeadingInfo(level=level, text=text, is_numbered=True)

        # 2. Visual heading (no numbering)
        if self._is_visual_heading(line):
            level = self._visual_level(line)
            return HeadingInfo(level=level, text=text, is_numbered=False)

        return None


# ============================================================
# HIERARCHY TRACKING
# ============================================================

class HierarchyTracker:
    """
    Maintains a stack of current headings.
    When a new heading appears:
      - If its level is <= the level of the last heading in the stack,
        pop until the stack's last heading has a lower level.
      - Then append the new heading.
    This gives a flexible heading path (e.g., ["Policy", "Scope"]).
    """

    def __init__(self):
        self.stack: List[Tuple[int, str]] = []  # (level, heading_text)

    def update(self, heading: HeadingInfo):
        level = heading.level
        text = heading.text

        # Pop deeper headings
        while self.stack and level <= self.stack[-1][0]:
            self.stack.pop()
        self.stack.append((level, text))

    @property
    def path(self) -> List[str]:
        return [text for _, text in self.stack]


# ============================================================
# CHUNK CREATION
# ============================================================

def split_long_text(text: str, max_chars: int) -> List[str]:
    """
    Split a large text block into smaller chunks, trying to break on
    paragraph boundaries first, then line boundaries.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        chunks = []
        current = []
        current_len = 0
        for para in paragraphs:
            if current and current_len + len(para) + 2 > max_chars:
                chunks.append("\n\n".join(current))
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para) + 2
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    # Fallback: split by single lines
    lines = text.split("\n")
    chunks = []
    current = []
    current_len = 0
    for line in lines:
        if current and current_len + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def create_chunks(
    lines: List[Line],
    detector: HeadingDetector,
    document_id: str,
    start_new_chunk_on_level: int = 1,  # start new chunk on headings of level <= this
    max_chars: int = 2500,
) -> List[Chunk]:
    """
    Create chunks while tracking the heading path.
    A new chunk begins when a heading of level <= `start_new_chunk_on_level` appears.
    """
    tracker = HierarchyTracker()
    chunks: List[Chunk] = []

    current_text: List[str] = []
    current_pages: List[int] = []
    current_path: List[str] = []

    chunk_counter = 1

    def save_current():
        nonlocal chunk_counter
        if not current_text:
            return
        text = "\n".join(current_text).strip()
        if not text:
            return

        for sub_text in split_long_text(text, max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_{chunk_counter:04d}",
                    text=sub_text,
                    page_start=min(current_pages),
                    page_end=max(current_pages),
                    heading_path=current_path.copy(),
                )
            )
            chunk_counter += 1

    for line in lines:
        heading = detector.detect(line)

        if heading:
            # Update hierarchy
            tracker.update(heading)

            # Decide whether to start a new chunk
            if heading.level <= start_new_chunk_on_level:
                save_current()
                current_text.clear()
                current_pages.clear()
                current_path = tracker.path.copy()

        # Add the line to the current chunk
        current_text.append(line.text)
        current_pages.append(line.page)

    save_current()
    return chunks


# ============================================================
# COMPLETE DOCUMENT PROCESSING
# ============================================================

def process_document(
    pdf_path: str,
    document_id: str,
    start_new_chunk_on_level: int = 1,
    max_chars: int = 2500,
) -> Dict[str, Any]:
    """
    Full pipeline: extract lines → detect headings → create chunks.
    """
    lines = extract_lines(pdf_path)
    if not lines:
        raise ValueError(f"No text extracted from {pdf_path}")

    detector = HeadingDetector(lines)
    chunks = create_chunks(
        lines,
        detector,
        document_id,
        start_new_chunk_on_level=start_new_chunk_on_level,
        max_chars=max_chars,
    )

    return {
        "document_id": document_id,
        "source_path": str(pdf_path),
        "element_count": len(lines),
        "chunk_count": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
    }


def process_documents(
    doc1: str,
    doc2: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Process two documents with the same settings.
    """
    return {
        "document_a": process_document(doc1, "document_a", **kwargs),
        "document_b": process_document(doc2, "document_b", **kwargs),
    }
