"""
Parsers for HTS tariff schedule PDFs and CROSS ruling PDFs.

Extracts structured chunks suitable for RAG ingestion.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

import pdfplumber


@dataclass
class Chunk:
    """A single text chunk for embedding and retrieval."""
    text: str
    source: str          # e.g. "hts_chapter_85" or "cross_N352071"
    chunk_type: str      # "hts_note", "hts_heading", "hts_subheading", "cross_ruling"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTS Parser
# ---------------------------------------------------------------------------

def parse_hts_chapter(pdf_path: Path) -> list[Chunk]:
    """Parse an HTS chapter PDF into chunks.

    Produces two kinds of chunks:
    1. Chapter notes / additional notes (full text, chunked by ~1500 chars)
    2. Heading-level groups (heading description + all subheadings under it)
    """
    chapter_num = _extract_chapter_number(pdf_path)
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    if not full_text.strip():
        return chunks

    # Split into notes section and tariff schedule section
    # The tariff schedule starts with the first 4-digit heading pattern
    heading_pattern = re.compile(r'^(\d{4})(?:\.\d{2})?(?:\.\d{2,4})?\s', re.MULTILINE)
    first_heading_match = heading_pattern.search(full_text)

    if first_heading_match:
        notes_text = full_text[:first_heading_match.start()]
        schedule_text = full_text[first_heading_match.start():]
    else:
        notes_text = full_text
        schedule_text = ""

    # Chunk notes (split on double newlines, merge small chunks)
    if notes_text.strip():
        note_chunks = _chunk_text(notes_text.strip(), max_chars=1500)
        for i, nc in enumerate(note_chunks):
            chunks.append(Chunk(
                text=f"HTS Chapter {chapter_num} Notes (part {i+1}):\n{nc}",
                source=f"hts_chapter_{chapter_num}",
                chunk_type="hts_note",
                metadata={"chapter": chapter_num, "part": i + 1},
            ))

    # Parse tariff schedule into heading groups
    if schedule_text.strip():
        heading_chunks = _parse_hts_schedule(schedule_text, chapter_num)
        chunks.extend(heading_chunks)

    return chunks


def _extract_chapter_number(pdf_path: Path) -> str:
    """Extract chapter number from filename like 'chapter_85.pdf'."""
    m = re.search(r'chapter_(\d+)', pdf_path.stem)
    return m.group(1) if m else pdf_path.stem


def _parse_hts_schedule(text: str, chapter_num: str) -> list[Chunk]:
    """Parse the tariff schedule portion into heading-level chunks."""
    chunks = []

    # Split by 4-digit heading boundaries
    # Pattern: start of line, 4-digit code followed by description
    lines = text.split('\n')
    current_heading = None
    current_lines = []

    for line in lines:
        # Detect a new heading (4-digit, e.g. "8501" at line start)
        heading_match = re.match(r'^(\d{4})\s+(.+)', line)
        if heading_match and heading_match.group(1)[:2] == chapter_num[:2]:
            # Save previous heading group
            if current_heading and current_lines:
                chunk_text = '\n'.join(current_lines)
                if len(chunk_text.strip()) > 50:
                    chunks.append(Chunk(
                        text=f"HTS Heading {current_heading}:\n{chunk_text}",
                        source=f"hts_chapter_{chapter_num}",
                        chunk_type="hts_heading",
                        metadata={
                            "chapter": chapter_num,
                            "heading": current_heading,
                        },
                    ))
            current_heading = heading_match.group(1)
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save last heading group
    if current_heading and current_lines:
        chunk_text = '\n'.join(current_lines)
        if len(chunk_text.strip()) > 50:
            chunks.append(Chunk(
                text=f"HTS Heading {current_heading}:\n{chunk_text}",
                source=f"hts_chapter_{chapter_num}",
                chunk_type="hts_heading",
                metadata={
                    "chapter": chapter_num,
                    "heading": current_heading,
                },
            ))

    # If headings are too large, split them further
    final_chunks = []
    for c in chunks:
        if len(c.text) > 3000:
            sub_chunks = _chunk_text(c.text, max_chars=2000)
            for i, sc in enumerate(sub_chunks):
                final_chunks.append(Chunk(
                    text=sc,
                    source=c.source,
                    chunk_type=c.chunk_type,
                    metadata={**c.metadata, "part": i + 1},
                ))
        else:
            final_chunks.append(c)

    return final_chunks


def parse_gri(pdf_path: Path) -> list[Chunk]:
    """Parse General Rules of Interpretation PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    if not full_text.strip():
        return []

    chunks = _chunk_text(full_text.strip(), max_chars=1500)
    return [
        Chunk(
            text=f"General Rules of Interpretation (part {i+1}):\n{c}",
            source="hts_gri",
            chunk_type="hts_note",
            metadata={"type": "gri", "part": i + 1},
        )
        for i, c in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# CROSS Ruling Parser
# ---------------------------------------------------------------------------

def parse_cross_ruling(pdf_path: Path) -> list[Chunk]:
    """Parse a single CROSS ruling PDF into one or more chunks."""
    ruling_number = pdf_path.stem  # e.g. "N352071"

    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    if not full_text.strip() or len(full_text.strip()) < 100:
        return []

    # Extract key metadata from ruling text
    tariff_match = re.search(r'TARIFF NO\.?:?\s*([\d\.]+)', full_text)
    tariff_no = tariff_match.group(1) if tariff_match else ""

    date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', full_text)
    ruling_date = date_match.group(0) if date_match else ""

    # Extract the RE: line (product description)
    re_match = re.search(r'RE:\s*(.+?)(?:\n(?:Dear|This))', full_text, re.DOTALL)
    product_desc = re_match.group(1).strip() if re_match else ""

    # Extract all tariff codes mentioned
    all_tariffs = re.findall(r'\b(\d{4}\.\d{2}(?:\.\d{2,4})?)\b', full_text)
    unique_tariffs = list(dict.fromkeys(all_tariffs))  # preserve order, deduplicate

    metadata = {
        "ruling_number": ruling_number,
        "tariff_no": tariff_no,
        "ruling_date": ruling_date,
        "product_description": product_desc,
        "all_tariff_codes": unique_tariffs,
        "collection": "NY" if ruling_number.startswith("N") else "HQ",
    }

    # For short rulings, keep as single chunk
    if len(full_text) <= 3000:
        return [Chunk(
            text=f"CROSS Ruling {ruling_number} (Tariff: {tariff_no}):\n{full_text.strip()}",
            source=f"cross_{ruling_number}",
            chunk_type="cross_ruling",
            metadata=metadata,
        )]

    # For long rulings, split meaningfully
    chunks = _chunk_text(full_text.strip(), max_chars=2000)
    return [
        Chunk(
            text=f"CROSS Ruling {ruling_number} (Tariff: {tariff_no}) [part {i+1}/{len(chunks)}]:\n{c}",
            source=f"cross_{ruling_number}",
            chunk_type="cross_ruling",
            metadata={**metadata, "part": i + 1},
        )
        for i, c in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chars: int = 1500) -> list[str]:
    """Split text into chunks, trying to break on paragraph boundaries."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    # Handle paragraphs that are themselves too long
    final = []
    for c in chunks:
        if len(c) > max_chars * 1.5:
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', c)
            sub = ""
            for s in sentences:
                if len(sub) + len(s) + 1 > max_chars and sub:
                    final.append(sub.strip())
                    sub = s
                else:
                    sub = sub + " " + s if sub else s
            if sub.strip():
                final.append(sub.strip())
        else:
            final.append(c)

    return final


def parse_all_hts(hts_dir: Path) -> list[Chunk]:
    """Parse all HTS PDFs in a directory."""
    chunks = []
    for pdf_path in sorted(hts_dir.glob("chapter_*.pdf")):
        print(f"  Parsing {pdf_path.name}...")
        parsed = parse_hts_chapter(pdf_path)
        chunks.extend(parsed)
        print(f"    -> {len(parsed)} chunks")

    # GRI
    gri_path = hts_dir / "general_rules_of_interpretation.pdf"
    if gri_path.exists():
        print(f"  Parsing GRI...")
        gri_chunks = parse_gri(gri_path)
        chunks.extend(gri_chunks)
        print(f"    -> {len(gri_chunks)} chunks")

    return chunks


def parse_all_cross(cross_dir: Path) -> list[Chunk]:
    """Parse all CROSS ruling PDFs."""
    chunks = []
    for collection in ["ny", "hq"]:
        col_dir = cross_dir / collection
        if not col_dir.exists():
            continue
        pdfs = sorted(col_dir.glob("*.pdf"))
        print(f"  Parsing {len(pdfs)} {collection.upper()} rulings...")
        for pdf_path in pdfs:
            try:
                parsed = parse_cross_ruling(pdf_path)
                chunks.extend(parsed)
            except Exception as e:
                print(f"    Error parsing {pdf_path.name}: {e}")
    return chunks
