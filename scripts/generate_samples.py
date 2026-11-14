"""Generate synthetic shipment PDFs (all data is fictional).

    python scripts/generate_samples.py

Creates data/samples/<case>/{commercial_invoice,packing_list,bill_of_lading}.pdf plus an
expected.json (the ground-truth issue codes, used by the tests -- the pipeline never reads it).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent.parent / "data" / "samples"
SS = getSampleStyleSheet()

BASE = {
    "invoice": {
        "invoice_no": "INV-2026-0417", "date": "12 Oct 2026",
        "seller": "Siam Seafood Export Co., Ltd.", "buyer": "Pacific Fresh Foods Inc.",
        "incoterm": "FOB Laem Chabang", "currency": "USD",
        "items": [  # description, hs, qty, unit price
            ["Frozen Shrimp 16/20", "0306.17", 1000, 8.50],
            ["Frozen Squid Rings", "0307.43", 600, 6.00],
            ["Canned Tuna in Oil", "1604.14", 2000, 1.25],
        ],
        "amount_override": {},       # line index -> wrong amount
        "total_override": None,
        "labels": {},                # field -> alternative label printed on the PDF
    },
    "packing_list": {
        "pl_no": "PL-2026-0417", "date": "13 Oct 2026", "invoice_ref": "INV-2026-0417",
        "items": [  # description, qty, packages, net kg, gross kg
            ["Frozen Shrimp 16/20", 1000, 100, 1000.0, 1090.0],
            ["Frozen Squid Rings", 600, 60, 600.0, 650.0],
            ["Canned Tuna in Oil", 2000, 40, 360.0, 410.0],
        ],
        "total_packages": 200, "gross": 2150.0, "net": 1960.0,
    },
    "bill_of_lading": {
        "bl_no": "BKKA260417", "shipper": "Siam Seafood Export Co., Ltd.",
        "consignee": "Pacific Fresh Foods Inc.", "vessel": "MV Andaman Star V.214",
        "port_loading": "Laem Chabang, Thailand", "port_discharge": "Los Angeles, USA",
        "packages": 200, "gross": 2150.0, "on_board": "15 Oct 2026",
        "container": "TGHU1234567",
    },
}


def money(x: float) -> str:
    return f"{x:,.2f}"


def build(path: Path, title: str, lines: list[str], table=None, footer=()):
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=title)
    story = [Paragraph(f"<b>{title}</b>", SS["Title"]), Spacer(1, 8)]
    story += [Paragraph(line, SS["Normal"]) for line in lines]
    if table:
        story.append(Spacer(1, 10))
        t = Table(table, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                               ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(t)
    story.append(Spacer(1, 10))
    story += [Paragraph(line, SS["Normal"]) for line in footer]
    doc.build(story)


def render_invoice(d: dict, path: Path):
    rows = [["No", "Description", "HS Code", "Qty", "Unit Price (USD)", "Amount (USD)"]]
    total = 0.0
    for i, (desc, hs, qty, price) in enumerate(d["items"]):
        amount = d["amount_override"].get(i, round(qty * price, 2))
        total += amount
        rows.append([str(i + 1), desc, hs, f"{qty:,}", money(price), money(amount)])
    correct_total = round(sum(q * p for _, _, q, p in d["items"]), 2)
    total_shown = d["total_override"] if d["total_override"] is not None else correct_total
    lab = {"invoice_no": "Invoice No", "buyer": "Buyer", **d["labels"]}
    build(path, "COMMERCIAL INVOICE", [
        f"{lab['invoice_no']}: {d['invoice_no']}", f"Invoice Date: {d['date']}",
        f"Seller: {d['seller']}", f"{lab['buyer']}: {d['buyer']}",
        f"Incoterms: {d['incoterm']}", f"Currency: {d['currency']}"],
        rows, [f"Total Amount: {money(total_shown)}"])


def render_pl(d: dict, path: Path):
    rows = [["No", "Description", "Qty", "Packages", "Net Wt (kg)", "Gross Wt (kg)"]]
    for i, (desc, qty, pk, net, gross) in enumerate(d["items"]):
        rows.append([str(i + 1), desc, f"{qty:,}", str(pk), money(net), money(gross)])
    build(path, "PACKING LIST", [
        f"Packing List No: {d['pl_no']}", f"Date: {d['date']}", f"Invoice No: {d['invoice_ref']}"],
        rows, [f"Total Packages: {d['total_packages']}",
               f"Total Gross Weight (kg): {money(d['gross'])}",
               f"Total Net Weight (kg): {money(d['net'])}"])


def render_bl(d: dict, path: Path):
    build(path, "BILL OF LADING", [
        f"B/L No: {d['bl_no']}", f"Shipper: {d['shipper']}", f"Consignee: {d['consignee']}",
        f"Vessel: {d['vessel']}", f"Port of Loading: {d['port_loading']}",
        f"Port of Discharge: {d['port_discharge']}", f"Container No: {d['container']}",
        f"Number of Packages: {d['packages']}", f"Gross Weight (kg): {money(d['gross'])}",
        f"Shipped on Board Date: {d['on_board']}"])


def case_clean():
    return copy.deepcopy(BASE), []


def case_discrepancies():
    d = copy.deepcopy(BASE)
    d["packing_list"]["items"][0][1] = 900                       # qty 1000 vs 900
    d["invoice"]["amount_override"] = {2: 2600.00}               # 2000*1.25 != 2600
    d["invoice"]["total_override"] = 14600.00                    # != sum of (wrong) lines
    d["bill_of_lading"]["shipper"] = "Siam Sea Food Exports Ltd."  # spelling variation
    d["bill_of_lading"]["gross"] = 2310.0                        # 7.4 % heavier than PL
    return d, ["QTY_MISMATCH", "INV_ARITHMETIC", "INV_TOTAL", "PARTY_SHIPPER", "GW_MISMATCH"]


def case_missing_bl():
    d = copy.deepcopy(BASE)
    d["invoice"]["items"][0][1] = "0306.1"                       # 5-digit HS code
    d["packing_list"]["invoice_ref"] = "INV-2026-0999"
    del d["bill_of_lading"]
    return d, ["MISSING_DOC", "HS_INVALID", "INVREF_MISMATCH"]


def case_dates_parties():
    d = copy.deepcopy(BASE)
    d["bill_of_lading"]["on_board"] = "05 Oct 2026"              # before invoice date
    d["bill_of_lading"]["packages"] = 190                        # vs 200
    d["bill_of_lading"]["consignee"] = "Oceanic Imports Inc."    # different party
    d["packing_list"]["net"] = 2200.0                            # net > gross
    return d, ["DATE_ORDER", "PKG_MISMATCH", "PARTY_CONSIGNEE", "NW_GT_GW"]


def case_unusual_labels():
    """Correct documents, but the invoice uses labels the rules do not know.

    Rule-based mode: fields unreadable -> REVIEW.
    With an LLM: the llm_repair node recovers them -> AUTO_PASS.
    """
    d = copy.deepcopy(BASE)
    d["invoice"]["labels"] = {"invoice_no": "Our Reference", "buyer": "Bill To"}
    return d, ["MISSING_FIELD"]


CASES = {
    "case_01_clean": case_clean,
    "case_02_discrepancies": case_discrepancies,
    "case_03_missing_bl": case_missing_bl,
    "case_04_dates_and_parties": case_dates_parties,
    "case_05_unusual_labels": case_unusual_labels,
}


def main():
    for name, fn in CASES.items():
        d, expected = fn()
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        render_invoice(d["invoice"], folder / "commercial_invoice.pdf")
        render_pl(d["packing_list"], folder / "packing_list.pdf")
        if "bill_of_lading" in d:
            render_bl(d["bill_of_lading"], folder / "bill_of_lading.pdf")
        (folder / "expected.json").write_text(
            json.dumps({"expected_codes": sorted(expected)}, indent=2))
        print("generated", folder.relative_to(OUT.parent.parent))


if __name__ == "__main__":
    main()
