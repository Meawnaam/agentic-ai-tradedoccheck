"""ExtractorAgent: turns document text/tables into structured fields.

Hybrid strategy:
  1. Fast, deterministic rules (label:value lines + line-item tables).
  2. If required fields are still missing and an LLM is configured, ask the LLM for ONLY the
     missing fields (self-correction step) and merge them in.
"""
from __future__ import annotations

import re
from datetime import datetime

from ..llm import BaseLLM, LLMError, extract_json
from ..models import REQUIRED_FIELDS

LABELS = {
    "invoice": {
        "invoice no": "invoice_no", "invoice number": "invoice_no",
        "invoice date": "date", "date": "date",
        "seller": "seller", "exporter": "seller",
        "buyer": "buyer", "sold to": "buyer",
        "incoterms": "incoterm", "incoterm": "incoterm",
        "currency": "currency", "total amount": "total_amount",
    },
    "packing_list": {
        "packing list no": "pl_no", "pl no": "pl_no", "date": "date",
        "invoice no": "invoice_ref", "invoice ref": "invoice_ref", "invoice reference": "invoice_ref",
        "total packages": "total_packages",
        "total gross weight kg": "gross_weight_kg", "total net weight kg": "net_weight_kg",
    },
    "bill_of_lading": {
        "b l no": "bl_no", "bl no": "bl_no", "bill of lading no": "bl_no",
        "shipper": "shipper", "consignee": "consignee", "vessel": "vessel",
        "port of loading": "port_loading", "port of discharge": "port_discharge",
        "number of packages": "packages", "gross weight kg": "gross_weight_kg",
        "shipped on board date": "on_board_date", "container no": "container_no",
    },
}
NUMERIC = {"total_amount", "total_packages", "gross_weight_kg", "net_weight_kg", "packages"}
DATES = {"date", "on_board_date"}
COLUMN_RULES = [  # (substring in normalised header, item key)
    ("unit price", "unit_price"), ("hs", "hs_code"), ("descr", "description"),
    ("amount", "amount"), ("qty", "qty"), ("quantity", "qty"),
    ("package", "packages"), ("net", "net_kg"), ("gross", "gross_kg"),
]
KV_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 .#/()]*?)\s*:\s*(.+?)\s*$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def parse_number(s) -> float | None:
    if s is None:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s))
    return float(m.group(0).replace(",", "")) if m else None


def parse_date(s: str) -> str | None:
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d/%m/%Y", "%d %B %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_items(tables, doc_type: str) -> list[dict]:
    for table in tables:
        if len(table) < 2:
            continue
        header = [_norm(h) for h in table[0]]
        if not any("descr" in h for h in header):
            continue
        cols: dict[int, str] = {}
        for i, h in enumerate(header):
            for sub, key in COLUMN_RULES:
                if sub in h:
                    cols[i] = key
                    break
        items = []
        for row in table[1:]:
            item = {}
            for i, key in cols.items():
                if i >= len(row):
                    continue
                item[key] = row[i] if key in ("description", "hs_code") else parse_number(row[i])
            if item.get("description"):
                items.append(item)
        return items
    return []


def rules_extract(doc_type: str, text: str, tables) -> dict:
    labels = LABELS[doc_type]
    fields: dict = {}
    for line in text.splitlines():
        m = KV_LINE.match(line)
        if not m:
            continue
        field = labels.get(_norm(m.group(1)))
        if not field or field in fields:
            continue
        value = m.group(2)
        if field in NUMERIC:
            fields[field] = parse_number(value)
        elif field in DATES:
            fields[field] = parse_date(value)
        else:
            fields[field] = value
    items = _parse_items(tables, doc_type)
    if items:
        fields["items"] = items
    return fields


def _is_empty(v) -> bool:
    return v is None or v == "" or v == []


def missing_fields(doc_type: str, fields: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS[doc_type] if _is_empty(fields.get(f))]


def _coerce(field: str, value):
    """Normalise a value returned by the LLM to the same types the rules produce."""
    if field in NUMERIC:
        return parse_number(value)
    if field in DATES and isinstance(value, str):
        return parse_date(value) or value
    return value


class ExtractorAgent:
    name = "ExtractorAgent"

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def extract_rules(self, doc_type: str, text: str, tables) -> tuple[dict, list[str]]:
        """Fast deterministic pass. Returns (fields, missing_required_fields)."""
        fields = rules_extract(doc_type, text, tables)
        return fields, missing_fields(doc_type, fields)

    def repair(self, doc_type: str, text: str, fields: dict, missing: list[str]) -> dict:
        """Self-correction: ask the LLM for ONLY the fields the rules could not find.

        Returns the fields it recovered (may be empty). Raises LLMError on failure.
        """
        reply = self.llm.complete(
            "You extract fields from international trade documents. Reply with ONE JSON object "
            "only, whose keys are exactly the requested field names. Use null when a field is "
            "truly not present -- never guess. Dates as YYYY-MM-DD, numbers as plain numbers. "
            "'items' is a list of objects with keys description, hs_code, qty, unit_price, "
            "amount, packages, net_kg, gross_kg. Labels may differ from the usual ones "
            "(e.g. 'Our Reference' can be the invoice number, 'Bill To' can be the buyer).",
            f"Document type: {doc_type}\nFields needed: {missing}\n\n--- DOCUMENT ---\n{text}",
            json_mode=True)
        data = extract_json(reply)
        recovered = {}
        for f in missing:
            v = data.get(f)
            if not _is_empty(v):
                recovered[f] = _coerce(f, v)
        return recovered

    def extract(self, doc_type: str, text: str, tables, trace) -> tuple[dict, list[str], str]:
        """Convenience: rules + (optional) one repair attempt. Returns (fields, missing, method)."""
        fields, missing = self.extract_rules(doc_type, text, tables)
        method = "rules"
        if missing and self.llm.available:
            trace("retry", f"rules missed {missing}; asking LLM for those fields only")
            try:
                fields.update(self.repair(doc_type, text, fields, missing))
                method = "rules+llm"
            except LLMError as e:
                trace("retry_failed", str(e))
            missing = missing_fields(doc_type, fields)
        return fields, missing, method
