"""InvestigatorAgent: an autonomous tool-using agent (ReAct-style loop).

Runs only for shipments the validator did not pass. The LLM receives the findings and a set of
tools, decides on its own which tools to call (and in what order) to confirm each problem and
find root causes, then calls `submit_report` with a summary, actions and a draft email.

Guardrails
  * The decision (AUTO_PASS / REVIEW / HOLD) is never changed by this agent.
  * Hard limit on steps; every tool call is logged and shown in the UI.
  * If no LLM is configured, or the LLM fails, a rule-based planner calls the same tools,
    so the demo always works and the output format is identical.
"""
from __future__ import annotations

import json
import os

from ..llm import BaseLLM, LLMError
from ..models import ExtractedDoc, Finding
from ..tools import TOOL_SPECS, CaseTools

MAX_STEPS = 12

SYSTEM = """You are a senior trade-documentation investigator at a Thai freight forwarder.
A deterministic validator has already compared the Commercial Invoice, Packing List and Bill of
Lading of one shipment and produced findings. Your job:
1. Use the tools to look at the evidence behind each finding (start with HIGH severity).
2. Work out the most likely root cause of each problem and how much money/weight is at stake.
3. Decide concrete next actions and who should do them (seller, shipper, forwarder, officer).
4. Finish by calling submit_report exactly once, including a short, polite email to the shipper
   (in English) listing every correction needed.
Rules: only state numbers that came from tool results; do not change the validator's decision;
do not call the same tool with the same arguments twice. Write `summary`, `root_causes` and
`actions` in {lang}."""


def _preview(result, limit: int = 600) -> str:
    s = json.dumps(result, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + " ..."


class InvestigatorAgent:
    name = "InvestigatorAgent"

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def investigate(self, docs: dict[str, ExtractedDoc], findings: list[Finding],
                    decision: str, lang: str | None = None) -> dict:
        lang = lang or os.getenv("TRADEDOC_LANG", "English")
        tools = CaseTools(docs, findings)
        if self.llm.available:
            try:
                return self._llm_loop(tools, findings, decision, lang)
            except LLMError as e:
                result = self._rule_plan(CaseTools(docs, findings), findings, decision)
                result["mode"] = f"rules (LLM failed: {str(e)[:120]})"
                return result
        return self._rule_plan(tools, findings, decision)

    # ------------------------------------------------------------------ LLM loop
    def _llm_loop(self, tools: CaseTools, findings, decision, lang) -> dict:
        steps: list[dict] = []
        prompt = (f"Validator decision: {decision}\n"
                  f"Findings ({len(findings)}):\n" +
                  "\n".join(f"- [{f.severity}] {f.code}: {f.message} "
                            f"(expected={f.expected}, actual={f.actual})" for f in findings) +
                  "\n\nInvestigate, then call submit_report.")
        conv = self.llm.start(SYSTEM.format(lang=lang), prompt, TOOL_SPECS)
        final_text = ""
        for _ in range(MAX_STEPS):
            calls, text = self.llm.step(conv)
            if text.strip():
                final_text = text.strip()
                steps.append({"type": "thought", "text": final_text})
            if not calls:
                break
            results = []
            for c in calls:
                r = tools.call(c.name, c.args)
                steps.append({"type": "tool", "tool": c.name, "args": c.args,
                              "result": _preview(r)})
                results.append((c, r))
            if tools.report is not None:
                break
            self.llm.add_results(conv, results)
        report = tools.report or {"summary": final_text or "Investigator stopped without a report.",
                                  "root_causes": [], "actions": [], "email_to_shipper": "",
                                  "escalate": decision == "HOLD"}
        return {"mode": f"llm ({self.llm.name}:{self.llm.model})", "steps": steps, **report}

    # ------------------------------------------------------------------ rule planner
    def _rule_plan(self, tools: CaseTools, findings: list[Finding], decision: str) -> dict:
        """Deterministic fallback: same tools, fixed plan per finding type."""
        steps: list[dict] = []
        done: set[str] = set()

        def use(name, **args):
            key = name + json.dumps(args, sort_keys=True)
            if key in done:
                return None
            done.add(key)
            r = tools.call(name, args)
            steps.append({"type": "tool", "tool": name, "args": args, "result": _preview(r)})
            return r

        causes, actions, corrections = [], [], []
        use("list_findings")
        for f in findings:
            if f.code == "MISSING_DOC":
                use("list_documents")
                doc = f.documents[0].replace("_", " ")
                causes.append(f"The {doc} was not received with the shipment set.")
                actions.append(f"Forwarder: obtain the {doc} before customs filing.")
                corrections.append(f"Please send the {doc} for this shipment.")
            elif f.code in ("INV_ARITHMETIC", "INV_TOTAL"):
                items = tools._f("invoice").get("items") or []
                for i, _ in enumerate(items, 1):
                    r = use("recheck_invoice_line", line_no=i)
                    if r and not r.get("consistent", True):
                        causes.append(f"Invoice line {i}: amount {r['amount']} but "
                                      f"qty x price = {r['computed_amount']}.")
                        corrections.append(f"Correct invoice line {i} amount to "
                                           f"{r['computed_amount']:,.2f}.")
                if f.code == "INV_TOTAL":
                    causes.append(f"Invoice total {f.actual} does not match the sum of line "
                                  f"amounts {f.expected}.")
                    corrections.append("Reissue the invoice with a corrected total.")
                actions.append("Seller: reissue the commercial invoice with corrected amounts.")
            elif f.code == "QTY_MISMATCH":
                desc = f.message.split("'")[1] if "'" in f.message else ""
                r = use("compare_items", description=desc) or {}
                stake = r.get("value_at_stake")
                causes.append(f"'{desc}': invoice qty {f.expected} vs packing list {f.actual}"
                              + (f" (value at stake {stake:,.2f})." if stake else "."))
                actions.append("Shipper: confirm the quantity actually loaded; seller to amend "
                               "invoice or packing list accordingly.")
                corrections.append(f"Confirm the shipped quantity of {desc}.")
            elif f.code in ("PARTY_SHIPPER", "PARTY_CONSIGNEE"):
                use("compare_parties")
                if f.severity == "LOW":
                    causes.append(f"Name spelled differently: '{f.expected}' vs '{f.actual}'.")
                    actions.append("Forwarder: request a B/L amendment if the L/C needs an "
                                   "exact name match.")
                else:
                    causes.append(f"Different party on the B/L: '{f.actual}' instead of "
                                  f"'{f.expected}'.")
                    actions.append("Officer: escalate -- possible wrong consignee; do not release "
                                   "documents.")
                corrections.append(f"Make the party name on the B/L read '{f.expected}'.")
            elif f.code in ("GW_MISMATCH", "PKG_MISMATCH", "PL_SUM", "NW_GT_GW"):
                r = use("compare_quantities") or {}
                if f.code == "GW_MISMATCH":
                    causes.append(f"Gross weight differs by {r.get('gross_weight_diff_kg')} kg "
                                  f"({r.get('gross_weight_diff_pct')}%).")
                    actions.append("Shipper: provide the VGM / weighbridge certificate.")
                    corrections.append("Confirm the verified gross mass and align the packing "
                                       "list and B/L.")
                elif f.code == "PKG_MISMATCH":
                    causes.append(f"Package count: packing list {f.expected} vs B/L {f.actual}.")
                    actions.append("Forwarder: arrange a tally / recount at the terminal.")
                    corrections.append("Confirm the number of packages loaded.")
                elif f.code == "NW_GT_GW":
                    causes.append("Net weight above gross weight -- likely a typing error.")
                    actions.append("Shipper: correct the packing list weights.")
                    corrections.append("Correct the net/gross weights on the packing list.")
                else:
                    causes.append("Packing list item packages do not sum to the total.")
                    corrections.append("Correct the package totals on the packing list.")
            elif f.code == "DATE_ORDER":
                use("check_dates")
                causes.append(f"On-board date {f.actual} is before the invoice date.")
                actions.append("Seller: confirm the invoice date (back-dated or typo).")
                corrections.append("Confirm the invoice date and on-board date.")
            elif f.code == "HS_INVALID":
                r = use("lookup_hs_code", code=str(f.actual)) or {}
                causes.append(f"HS code {f.actual} has {r.get('digits')} digits "
                              f"(chapter: {r.get('chapter')}).")
                actions.append("Officer: confirm the full HS code before the customs declaration.")
                corrections.append(f"Provide the full HS code for the item coded {f.actual}.")
            elif f.code == "INVREF_MISMATCH":
                use("get_document", doc_type="packing_list")
                causes.append(f"Packing list refers to invoice {f.actual}, not {f.expected}.")
                actions.append("Shipper: confirm the packing list belongs to this shipment.")
                corrections.append(f"Reissue the packing list referencing invoice {f.expected}.")
            elif f.code in ("MISSING_FIELD", "ITEM_UNMATCHED"):
                causes.append(f.message)
                actions.append("Officer: check the source PDF manually.")

        n_high = sum(f.severity == "HIGH" for f in findings)
        summary = (f"Decision {decision}: {len(findings)} issue(s), {n_high} high severity. "
                   + ("Documents must be corrected before release." if n_high else
                      "Minor issues; a reviewer should confirm before filing."))
        email = ""
        if corrections:
            email = ("Subject: Document corrections required for your shipment\n\n"
                     "Dear Shipper,\n\nDuring our document check we found the following items "
                     "that need your confirmation or correction:\n\n" +
                     "\n".join(f"{i}. {c}" for i, c in enumerate(dict.fromkeys(corrections), 1)) +
                     "\n\nPlease send the corrected documents at your earliest convenience so we "
                     "can proceed with customs clearance.\n\nBest regards,\nTrade Documentation "
                     "Team")
        tools.call("submit_report", {"summary": summary, "root_causes": causes,
                                     "actions": list(dict.fromkeys(actions)),
                                     "email_to_shipper": email, "escalate": n_high > 0})
        steps.append({"type": "tool", "tool": "submit_report", "args": {"summary": summary},
                      "result": '{"status": "report received"}'})
        return {"mode": "rules", "steps": steps, **tools.report}
