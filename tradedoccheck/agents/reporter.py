"""ReporterAgent: turns findings into a decision and a human-readable report."""
from __future__ import annotations

import json
import os

from ..llm import BaseLLM, LLMError
from ..models import CaseResult, Finding


def decide(findings: list[Finding]) -> str:
    if any(f.severity == "HIGH" for f in findings):
        return "HOLD"
    return "REVIEW" if findings else "AUTO_PASS"


class ReporterAgent:
    name = "ReporterAgent"

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def summarise(self, decision: str, findings: list[Finding]) -> str:
        n_high = sum(f.severity == "HIGH" for f in findings)
        template = {
            "AUTO_PASS": "All cross-document checks passed. No manual review needed.",
            "REVIEW": f"{len(findings)} minor issue(s) found. A reviewer should confirm before filing.",
            "HOLD": f"{n_high} serious discrepancy(ies) found ({len(findings)} issues total). "
                    "Hold the shipment documents until resolved.",
        }[decision]
        if not (self.llm.available and findings):
            return template
        lang = os.getenv("TRADEDOC_LANG", "English")
        try:
            payload = json.dumps([f.__dict__ for f in findings], ensure_ascii=False)
            reply = self.llm.complete(
                f"You write a 2-3 sentence summary for a customs/logistics officer in {lang}. "
                "Use ONLY the findings provided; do not invent facts or numbers.",
                f"Decision: {decision}\nFindings JSON: {payload}")
            return reply.strip() or template
        except LLMError:
            return template

    @staticmethod
    def to_markdown(r: CaseResult) -> str:
        icon = {"AUTO_PASS": "PASS", "REVIEW": "REVIEW", "HOLD": "HOLD"}[r.decision]
        lines = [f"# Case `{r.case_id}` -- {icon}", "", r.summary, "", "## Documents read", ""]
        for d in r.documents:
            miss = f" (missing: {', '.join(d.missing)})" if d.missing else ""
            lines.append(f"- **{d.doc_type}** `{d.source}` via {d.extraction_method}{miss}")
        lines += ["", "## Findings", ""]
        if not r.findings:
            lines.append("_None._")
        else:
            lines += ["| Severity | Code | Issue | Expected | Actual | Suggested action |",
                      "|---|---|---|---|---|---|"]
            for f in r.findings:
                lines.append(f"| {f.severity} | {f.code} | {f.message} | "
                             f"{'' if f.expected is None else f.expected} | "
                             f"{'' if f.actual is None else f.actual} | {f.suggestion} |")
        inv = r.investigation
        if inv:
            lines += ["", f"## Investigation ({inv.get('mode')})", "", inv.get("summary", ""), ""]
            if inv.get("root_causes"):
                lines += ["**Root causes**", ""] + [f"- {c}" for c in inv["root_causes"]] + [""]
            if inv.get("actions"):
                lines += ["**Actions**", ""] + [f"1. {a}" for a in inv["actions"]] + [""]
            tools = [s["tool"] for s in inv.get("steps", []) if s.get("type") == "tool"]
            lines += [f"**Tools called by the agent ({len(tools)}):** " + " > ".join(tools), ""]
            if inv.get("email_to_shipper"):
                lines += ["**Draft email to shipper**", "", "```text", inv["email_to_shipper"],
                          "```"]
        lines += ["", "## Workflow", "", f"Engine: {r.engine} | LLM: {r.llm} | Path: "
                  + " -> ".join(r.path), "", "## Agent trace", ""]
        lines += [f"1. `{t['agent']}` {t['action']}: {t['detail']}" for t in r.trace]
        return "\n".join(lines) + "\n"
