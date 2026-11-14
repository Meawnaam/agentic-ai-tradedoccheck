"""Tools the InvestigatorAgent can call.

Each tool is a plain Python function over the current case (extracted documents + findings),
plus a JSON-schema description that is sent to the LLM. The LLM chooses which tools to call and
in what order; the tools themselves are deterministic, so every number the agent reports can be
traced back to a tool result.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import asdict
from typing import Any, Callable

from .agents.validator import norm_name, similarity
from .llm import ToolSpec
from .models import DOC_TYPES, ExtractedDoc, Finding

# Small reference table of real HS chapters/headings (subset, for demo purposes).
HS_REFERENCE = {
    "03": "Fish and crustaceans, molluscs and other aquatic invertebrates",
    "0303": "Fish, frozen",
    "0306": "Crustaceans (e.g. shrimps, prawns, crabs), live/fresh/chilled/frozen",
    "0307": "Molluscs (e.g. squid, octopus, oysters), live/fresh/chilled/frozen",
    "04": "Dairy produce; birds' eggs; natural honey",
    "08": "Edible fruit and nuts",
    "10": "Cereals",
    "1006": "Rice",
    "16": "Preparations of meat, fish, crustaceans or molluscs",
    "1604": "Prepared or preserved fish (e.g. canned tuna)",
    "1605": "Crustaceans and molluscs, prepared or preserved",
    "17": "Sugars and sugar confectionery",
    "22": "Beverages, spirits and vinegar",
    "27": "Mineral fuels, mineral oils and products of their distillation",
    "2709": "Petroleum oils, crude",
    "39": "Plastics and articles thereof",
    "40": "Rubber and articles thereof",
    "61": "Apparel, knitted or crocheted",
    "62": "Apparel, not knitted or crocheted",
    "84": "Machinery and mechanical appliances",
    "85": "Electrical machinery and equipment",
    "87": "Vehicles other than railway",
    "94": "Furniture; bedding; lamps",
}


class CaseTools:
    """Binds the tool functions to one case."""

    def __init__(self, docs: dict[str, ExtractedDoc], findings: list[Finding]):
        self.docs, self.findings = docs, findings
        self.report: dict | None = None  # filled by submit_report

    def _f(self, doc_type: str) -> dict:
        return self.docs[doc_type].fields if doc_type in self.docs else {}

    # ---------------------------------------------------------------- read-only tools
    def list_documents(self) -> dict:
        return {"present": [{"doc_type": d.doc_type, "file": d.source,
                             "extraction": d.extraction_method, "unreadable_fields": d.missing}
                            for d in self.docs.values()],
                "absent": [t for t in DOC_TYPES if t not in self.docs]}

    def get_document(self, doc_type: str) -> dict:
        if doc_type not in self.docs:
            return {"error": f"{doc_type} not in this case", "available": list(self.docs)}
        return self.docs[doc_type].fields

    def list_findings(self, severity: str = "") -> list[dict]:
        out = []
        for i, f in enumerate(self.findings):
            if severity and f.severity != severity.upper():
                continue
            d = asdict(f)
            d["index"] = i
            out.append(d)
        return out

    def recheck_invoice_line(self, line_no: int) -> dict:
        items = self._f("invoice").get("items") or []
        line_no = int(line_no)
        if not 1 <= line_no <= len(items):
            return {"error": f"invoice has {len(items)} lines"}
        it = items[line_no - 1]
        calc = round((it.get("qty") or 0) * (it.get("unit_price") or 0), 2)
        return {"line": line_no, **it, "computed_amount": calc,
                "difference": round((it.get("amount") or 0) - calc, 2),
                "consistent": abs((it.get("amount") or 0) - calc) <= 0.01,
                "invoice_total_shown": self._f("invoice").get("total_amount"),
                "sum_of_shown_line_amounts": round(sum(i.get("amount") or 0 for i in items), 2),
                "sum_of_computed_line_amounts": round(sum((i.get("qty") or 0) *
                                                          (i.get("unit_price") or 0)
                                                          for i in items), 2)}

    def compare_items(self, description: str) -> dict:
        def best(items):
            scored = [(difflib.SequenceMatcher(None, norm_name(description),
                                               norm_name(i.get("description", ""))).ratio(), i)
                      for i in items]
            return max(scored, key=lambda x: x[0], default=(0, None))
        s1, inv = best(self._f("invoice").get("items") or [])
        s2, pl = best(self._f("packing_list").get("items") or [])
        result = {"query": description,
                  "invoice_item": inv if s1 >= 0.6 else None,
                  "packing_list_item": pl if s2 >= 0.6 else None}
        if inv and pl and s1 >= 0.6 and s2 >= 0.6:
            q1, q2 = inv.get("qty"), pl.get("qty")
            result["qty_difference"] = None if q1 is None or q2 is None else q2 - q1
            if inv.get("unit_price") and result["qty_difference"]:
                result["value_at_stake"] = round(abs(result["qty_difference"]) *
                                                 inv["unit_price"], 2)
        return result

    def compare_parties(self) -> dict:
        inv, bl = self._f("invoice"), self._f("bill_of_lading")
        out = {}
        for a, b in (("seller", "shipper"), ("buyer", "consignee")):
            va, vb = inv.get(a), bl.get(b)
            out[f"{a}_vs_{b}"] = {"invoice": va, "bill_of_lading": vb,
                                  "similarity": round(similarity(va, vb), 3) if va and vb else None}
        return out

    def compare_quantities(self) -> dict:
        pl, bl = self._f("packing_list"), self._f("bill_of_lading")
        items = pl.get("items") or []
        out = {"packing_list": {"total_packages": pl.get("total_packages"),
                                "gross_weight_kg": pl.get("gross_weight_kg"),
                                "net_weight_kg": pl.get("net_weight_kg"),
                                "sum_item_packages": sum(i.get("packages") or 0 for i in items),
                                "sum_item_gross_kg": round(sum(i.get("gross_kg") or 0
                                                               for i in items), 2),
                                "sum_item_net_kg": round(sum(i.get("net_kg") or 0
                                                             for i in items), 2)},
               "bill_of_lading": {"packages": bl.get("packages"),
                                  "gross_weight_kg": bl.get("gross_weight_kg")}}
        a, b = pl.get("gross_weight_kg"), bl.get("gross_weight_kg")
        if a and b:
            out["gross_weight_diff_kg"] = round(b - a, 2)
            out["gross_weight_diff_pct"] = round((b - a) / a * 100, 2)
        return out

    def check_dates(self) -> dict:
        d = {"invoice_date": self._f("invoice").get("date"),
             "packing_list_date": self._f("packing_list").get("date"),
             "on_board_date": self._f("bill_of_lading").get("on_board_date")}
        seq = [v for v in d.values() if v]
        d["chronological"] = seq == sorted(seq)
        return d

    def lookup_hs_code(self, code: str) -> dict:
        digits = re.sub(r"\D", "", str(code))
        return {"code": code, "digits": len(digits),
                "valid_length": len(digits) in (6, 8, 10),
                "chapter": HS_REFERENCE.get(digits[:2], "unknown chapter"),
                "heading": HS_REFERENCE.get(digits[:4], "not in local reference table"),
                "note": "Local reference table only; confirm with the national customs tariff."}

    # ---------------------------------------------------------------- final answer
    def submit_report(self, summary: str, root_causes: list | None = None,
                      actions: list | None = None, email_to_shipper: str = "",
                      escalate: bool = False) -> dict:
        self.report = {"summary": summary, "root_causes": list(root_causes or []),
                       "actions": list(actions or []), "email_to_shipper": email_to_shipper,
                       "escalate": bool(escalate)}
        return {"status": "report received"}

    # ---------------------------------------------------------------- registry
    def registry(self) -> dict[str, Callable[..., Any]]:
        return {s.name: getattr(self, s.name) for s in TOOL_SPECS}

    def call(self, name: str, args: dict) -> Any:
        fn = self.registry().get(name)
        if fn is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}


def _obj(props: dict | None = None, required: list | None = None) -> dict:
    schema: dict = {"type": "object", "properties": props or {}}
    if required:
        schema["required"] = required
    return schema


TOOL_SPECS = [
    ToolSpec("list_documents", "List which documents were received and which are absent.", _obj()),
    ToolSpec("get_document", "Get all extracted fields of one document.",
             _obj({"doc_type": {"type": "string", "enum": list(DOC_TYPES)}}, ["doc_type"])),
    ToolSpec("list_findings", "List the validator's findings, optionally filtered by severity.",
             _obj({"severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}})),
    ToolSpec("recheck_invoice_line",
             "Recompute qty x unit price for one invoice line and compare invoice totals.",
             _obj({"line_no": {"type": "integer", "description": "1-based line number"}},
                  ["line_no"])),
    ToolSpec("compare_items",
             "Compare one item between invoice and packing list (qty, value at stake).",
             _obj({"description": {"type": "string"}}, ["description"])),
    ToolSpec("compare_parties", "Compare seller/shipper and buyer/consignee names.", _obj()),
    ToolSpec("compare_quantities",
             "Compare package counts and weights between packing list items, totals and B/L.",
             _obj()),
    ToolSpec("check_dates", "Check invoice, packing list and on-board dates are chronological.",
             _obj()),
    ToolSpec("lookup_hs_code", "Validate an HS code and look up its chapter/heading.",
             _obj({"code": {"type": "string"}}, ["code"])),
    ToolSpec("submit_report",
             "Submit the final investigation report. Call exactly once, at the end.",
             _obj({"summary": {"type": "string",
                               "description": "2-4 sentences for the documentation officer"},
                   "root_causes": {"type": "array", "items": {"type": "string"}},
                   "actions": {"type": "array", "items": {"type": "string"},
                               "description": "concrete next steps, most urgent first"},
                   "email_to_shipper": {"type": "string",
                                        "description": "draft email requesting corrections"},
                   "escalate": {"type": "boolean",
                                "description": "true if a supervisor must be involved"}},
                  ["summary", "actions"])),
]
