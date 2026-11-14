"""ValidatorAgent: deterministic cross-document checks.

This is intentionally NOT done by an LLM: pass/fail decisions on money, weights and quantities
must be reproducible and auditable.
"""
from __future__ import annotations

import difflib
import re

from ..models import DOC_TYPES, ExtractedDoc, Finding

STOP_WORDS = {"CO", "LTD", "LIMITED", "INC", "CORP", "CORPORATION", "COMPANY", "PCL",
              "PUBLIC", "LLC", "THE"}
WEIGHT_TOL = 0.005  # 0.5 %
HIGH_WEIGHT_DIFF = 0.02  # >2 % is HIGH


def norm_name(s: str) -> str:
    words = re.sub(r"[^A-Z0-9 ]", " ", s.upper()).split()
    return " ".join(w for w in words if w not in STOP_WORDS)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_name(a), norm_name(b)).ratio()


def _has(v) -> bool:
    return v not in (None, "", [])


class ValidatorAgent:
    name = "ValidatorAgent"

    def validate(self, docs: dict[str, ExtractedDoc]) -> list[Finding]:
        findings: list[Finding] = []
        for t in DOC_TYPES:
            if t not in docs:
                findings.append(Finding(
                    "MISSING_DOC", "HIGH", f"Required document missing: {t}", [t],
                    suggestion=f"Request the {t.replace('_', ' ')} from the shipper/forwarder."))
        for t, d in docs.items():
            for f in d.missing:
                findings.append(Finding(
                    "MISSING_FIELD", "MEDIUM", f"{t}: could not read field '{f}'", [t],
                    suggestion="Check the source PDF manually; it may be scanned or unusual."))

        inv, pl, bl = (docs[t].fields if t in docs else {} for t in DOC_TYPES)
        if inv:
            findings += self._invoice_internal(inv)
        if inv and pl:
            findings += self._invoice_vs_pl(inv, pl)
        if pl:
            findings += self._pl_internal(pl)
        if inv and bl:
            findings += self._invoice_vs_bl(inv, bl)
        if pl and bl:
            findings += self._pl_vs_bl(pl, bl)
        findings.sort(key=lambda f: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[f.severity])
        return findings

    # ---- single-document checks -------------------------------------------------------
    def _invoice_internal(self, inv) -> list[Finding]:
        out: list[Finding] = []
        items = inv.get("items") or []
        total = 0.0
        for i, it in enumerate(items, 1):
            if _has(it.get("qty")) and _has(it.get("unit_price")) and _has(it.get("amount")):
                calc = round(it["qty"] * it["unit_price"], 2)
                if abs(calc - it["amount"]) > 0.01:
                    out.append(Finding(
                        "INV_ARITHMETIC", "HIGH",
                        f"Invoice line {i} ({it.get('description')}): qty x unit price != amount",
                        ["invoice"], expected=calc, actual=it["amount"],
                        suggestion="Ask the seller to reissue the invoice with correct line amount."))
            total += it.get("amount") or 0
            hs = re.sub(r"\D", "", it.get("hs_code") or "")
            if it.get("hs_code") and len(hs) not in (6, 8, 10):
                out.append(Finding(
                    "HS_INVALID", "MEDIUM",
                    f"Invoice line {i}: HS code '{it['hs_code']}' has {len(hs)} digits "
                    "(expected 6, 8 or 10)", ["invoice"], actual=it["hs_code"],
                    suggestion="Confirm the correct HS classification before customs filing."))
        if items and _has(inv.get("total_amount")) and abs(total - inv["total_amount"]) > 0.01:
            out.append(Finding(
                "INV_TOTAL", "HIGH", "Invoice total does not equal the sum of its line amounts",
                ["invoice"], expected=round(total, 2), actual=inv["total_amount"],
                suggestion="Ask the seller to correct the invoice total."))
        return out

    def _pl_internal(self, pl) -> list[Finding]:
        out: list[Finding] = []
        items = pl.get("items") or []
        pk = sum(it.get("packages") or 0 for it in items)
        if items and _has(pl.get("total_packages")) and pk and pk != pl["total_packages"]:
            out.append(Finding(
                "PL_SUM", "MEDIUM", "Packing list: item packages do not add up to total packages",
                ["packing_list"], expected=pk, actual=pl["total_packages"]))
        if (_has(pl.get("net_weight_kg")) and _has(pl.get("gross_weight_kg"))
                and pl["net_weight_kg"] > pl["gross_weight_kg"]):
            out.append(Finding(
                "NW_GT_GW", "HIGH", "Packing list: net weight is greater than gross weight",
                ["packing_list"], expected=f"<= {pl['gross_weight_kg']}", actual=pl["net_weight_kg"],
                suggestion="Impossible weights; ask the shipper to correct the packing list."))
        return out

    # ---- cross-document checks --------------------------------------------------------
    def _invoice_vs_pl(self, inv, pl) -> list[Finding]:
        out: list[Finding] = []
        if _has(inv.get("invoice_no")) and _has(pl.get("invoice_ref")):
            if inv["invoice_no"].strip().upper() != pl["invoice_ref"].strip().upper():
                out.append(Finding(
                    "INVREF_MISMATCH", "HIGH",
                    "Packing list references a different invoice number",
                    ["invoice", "packing_list"], expected=inv["invoice_no"],
                    actual=pl["invoice_ref"],
                    suggestion="Verify the packing list belongs to this shipment."))
        pl_items = list(pl.get("items") or [])
        used: set[int] = set()
        for it in inv.get("items") or []:
            best, best_s = None, 0.0
            for j, p in enumerate(pl_items):
                if j in used:
                    continue
                s = difflib.SequenceMatcher(None, norm_name(it["description"]),
                                            norm_name(p["description"])).ratio()
                if s > best_s:
                    best, best_s = j, s
            if best is None or best_s < 0.6:
                out.append(Finding(
                    "ITEM_UNMATCHED", "HIGH",
                    f"Invoice item '{it['description']}' not found in packing list",
                    ["invoice", "packing_list"]))
                continue
            used.add(best)
            p = pl_items[best]
            if _has(it.get("qty")) and _has(p.get("qty")) and abs(it["qty"] - p["qty"]) > 1e-9:
                out.append(Finding(
                    "QTY_MISMATCH", "HIGH",
                    f"Quantity differs for '{it['description']}'",
                    ["invoice", "packing_list"], expected=it["qty"], actual=p["qty"],
                    suggestion="Confirm actual shipped quantity; short/over shipment affects payment."))
        for j, p in enumerate(pl_items):
            if j not in used:
                out.append(Finding(
                    "ITEM_UNMATCHED", "HIGH",
                    f"Packing list item '{p['description']}' not found in invoice",
                    ["invoice", "packing_list"]))
        return out

    def _invoice_vs_bl(self, inv, bl) -> list[Finding]:
        out: list[Finding] = []
        for a_key, b_key, code, label in (("seller", "shipper", "PARTY_SHIPPER", "Seller vs shipper"),
                                          ("buyer", "consignee", "PARTY_CONSIGNEE",
                                           "Buyer vs consignee")):
            a, b = inv.get(a_key), bl.get(b_key)
            if not (_has(a) and _has(b)):
                continue
            s = similarity(a, b)
            if s >= 0.999:
                continue
            sev = "LOW" if s >= 0.8 else "HIGH"
            out.append(Finding(
                code, sev, f"{label}: names differ (similarity {s:.0%})",
                ["invoice", "bill_of_lading"], expected=a, actual=b,
                suggestion=("Likely a spelling variation; confirm and amend the B/L if the L/C "
                            "requires exact match." if sev == "LOW" else
                            "Different party. Do not release documents until clarified.")))
        inv_d, bl_d = inv.get("date"), bl.get("on_board_date")
        if _has(inv_d) and _has(bl_d) and bl_d < inv_d:
            out.append(Finding(
                "DATE_ORDER", "MEDIUM", "Goods shipped on board before the invoice date",
                ["invoice", "bill_of_lading"], expected=f">= {inv_d}", actual=bl_d,
                suggestion="Check dates for typos; late-issued invoices can be a compliance flag."))
        return out

    def _pl_vs_bl(self, pl, bl) -> list[Finding]:
        out: list[Finding] = []
        if _has(pl.get("total_packages")) and _has(bl.get("packages")):
            if pl["total_packages"] != bl["packages"]:
                out.append(Finding(
                    "PKG_MISMATCH", "HIGH", "Number of packages differs between packing list and B/L",
                    ["packing_list", "bill_of_lading"], expected=pl["total_packages"],
                    actual=bl["packages"],
                    suggestion="Recount cartons at the port; a mismatch can block customs clearance."))
        a, b = pl.get("gross_weight_kg"), bl.get("gross_weight_kg")
        if _has(a) and _has(b) and a:
            diff = abs(a - b) / a
            if diff > WEIGHT_TOL:
                out.append(Finding(
                    "GW_MISMATCH", "HIGH" if diff > HIGH_WEIGHT_DIFF else "MEDIUM",
                    f"Gross weight differs between packing list and B/L ({diff:.1%})",
                    ["packing_list", "bill_of_lading"], expected=a, actual=b,
                    suggestion="Request a weighbridge / VGM certificate to confirm the true weight."))
        return out
