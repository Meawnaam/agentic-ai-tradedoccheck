"""Shared data models."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

DOC_TYPES = ("invoice", "packing_list", "bill_of_lading")

# Fields every document type must provide for the cross-checks to work.
REQUIRED_FIELDS = {
    "invoice": ["invoice_no", "date", "seller", "buyer", "currency", "total_amount", "items"],
    "packing_list": ["pl_no", "invoice_ref", "total_packages", "gross_weight_kg",
                     "net_weight_kg", "items"],
    "bill_of_lading": ["bl_no", "shipper", "consignee", "port_loading", "port_discharge",
                       "packages", "gross_weight_kg", "on_board_date"],
}

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


@dataclass
class ExtractedDoc:
    doc_type: str
    source: str
    fields: dict[str, Any]
    missing: list[str] = field(default_factory=list)
    extraction_method: str = "rules"


@dataclass
class Finding:
    code: str
    severity: str  # HIGH | MEDIUM | LOW
    message: str
    documents: list[str]
    expected: Any = None
    actual: Any = None
    suggestion: str = ""


@dataclass
class CaseResult:
    case_id: str
    documents: list[ExtractedDoc]
    findings: list[Finding]
    decision: str  # AUTO_PASS | REVIEW | HOLD
    summary: str
    trace: list[dict]
    investigation: dict | None = None   # InvestigatorAgent output (REVIEW / HOLD only)
    path: list[str] = field(default_factory=list)   # graph nodes visited, in order
    engine: str = ""                    # "langgraph" or "builtin"
    llm: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)
