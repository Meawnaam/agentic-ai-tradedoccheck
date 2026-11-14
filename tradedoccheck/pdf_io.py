"""PDF reading. Digital (text-based) PDFs only; scanned PDFs would need an OCR step first."""
from __future__ import annotations

import pdfplumber


def read_pdf(path: str) -> tuple[str, list[list[list[str]]]]:
    """Return (full_text, tables). Each table is a list of rows of cell strings."""
    texts: list[str] = []
    tables: list[list[list[str]]] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
            for t in page.extract_tables():
                tables.append([[(c or "").strip() for c in row] for row in t])
    return "\n".join(texts), tables
