"""LangGraph workflow: one shipment folder in, one decision + report out.

    ingest -> classify -> extract --(fields missing & LLM on)--> llm_repair --+
                             |                                               |
                             +---------------(all fields found)--------------+--> validate
    validate -> decide --(AUTO_PASS)--> report -> END
                    +-----(REVIEW / HOLD)--> investigate (tool-using LLM agent) -> report -> END
"""
from __future__ import annotations

import operator
import os
from pathlib import Path
from typing import Annotated, Any, TypedDict

from .agents.classifier import ClassifierAgent
from .agents.extractor import ExtractorAgent, missing_fields
from .agents.investigator import InvestigatorAgent
from .agents.reporter import ReporterAgent, decide
from .agents.validator import ValidatorAgent
from .llm import BaseLLM, LLMError
from .models import ExtractedDoc
from .pdf_io import read_pdf

try:  # real LangGraph when installed (see requirements.txt)
    from langgraph.graph import END, START, StateGraph
    ENGINE = "langgraph"
except ImportError:  # offline fallback with the same semantics
    from ._minigraph import END, START, StateGraph
    ENGINE = "builtin"

MAX_REPAIR_ATTEMPTS = 1


class CaseState(TypedDict, total=False):
    case_id: str
    folder: str
    lang: str
    raw: list[dict]                       # [{source, text, tables, doc_type?}]
    docs: dict[str, ExtractedDoc]         # doc_type -> extracted document
    texts: dict[str, str]                 # doc_type -> raw text (for LLM repair)
    repair_attempts: int
    findings: list
    decision: str
    investigation: dict
    summary: str
    trace: Annotated[list[dict], operator.add]   # appended by every node
    path: Annotated[list[str], operator.add]     # names of nodes visited


# Static description of the graph, used by the UI to draw it.
NODES = ["ingest", "classify", "extract", "llm_repair", "validate", "decide", "investigate",
         "report"]
EDGES = [("ingest", "classify", ""), ("classify", "extract", ""),
         ("extract", "llm_repair", "fields missing"), ("extract", "validate", "complete"),
         ("llm_repair", "validate", ""), ("validate", "decide", ""),
         ("decide", "report", "AUTO_PASS"), ("decide", "investigate", "REVIEW / HOLD"),
         ("investigate", "report", "")]


def _t(agent: str, action: str, detail: Any) -> dict:
    return {"agent": agent, "action": action, "detail": detail}


def build_graph(llm: BaseLLM):
    classifier, extractor = ClassifierAgent(llm), ExtractorAgent(llm)
    validator, reporter, investigator = ValidatorAgent(), ReporterAgent(llm), InvestigatorAgent(llm)

    # ------------------------------------------------------------------ nodes
    def ingest(state: CaseState):
        raw, trace = [], []
        for pdf in sorted(Path(state["folder"]).glob("*.pdf")):
            try:
                text, tables = read_pdf(str(pdf))
            except Exception as e:  # corrupt / encrypted PDF
                trace.append(_t("Orchestrator", "read_failed", f"{pdf.name}: {e}"))
                continue
            raw.append({"source": pdf.name, "text": text, "tables": tables})
            trace.append(_t("Orchestrator", "read", f"{pdf.name}: {len(text)} chars, "
                                                    f"{len(tables)} table(s)"))
        return {"raw": raw, "trace": trace, "path": ["ingest"]}

    def classify(state: CaseState):
        raw, trace = [], []
        for r in state["raw"]:
            doc_type, how = classifier.classify(r["text"])
            raw.append({**r, "doc_type": doc_type})
            trace.append(_t("ClassifierAgent", "classify", f"{r['source']} -> {doc_type} ({how})"))
        return {"raw": raw, "trace": trace, "path": ["classify"]}

    def extract(state: CaseState):
        docs, texts, trace = {}, {}, []
        for r in state["raw"]:
            t = r["doc_type"]
            if t == "unknown":
                trace.append(_t("Orchestrator", "skip", f"{r['source']}: unknown type -> human"))
                continue
            if t in docs:
                trace.append(_t("Orchestrator", "skip", f"{r['source']}: duplicate {t}"))
                continue
            fields, missing = extractor.extract_rules(t, r["text"], r["tables"])
            docs[t] = ExtractedDoc(t, r["source"], fields, missing, "rules")
            texts[t] = r["text"]
            trace.append(_t("ExtractorAgent", "extract",
                            f"{r['source']}: {len(fields)} fields, missing={missing or 'none'}"))
        return {"docs": docs, "texts": texts, "trace": trace, "path": ["extract"]}

    def llm_repair(state: CaseState):
        docs, trace = dict(state["docs"]), []
        for t, d in docs.items():
            if not d.missing:
                continue
            trace.append(_t("ExtractorAgent", "llm_repair", f"{t}: asking LLM for {d.missing}"))
            try:
                recovered = extractor.repair(t, state["texts"][t], d.fields, d.missing)
            except LLMError as e:
                trace.append(_t("ExtractorAgent", "llm_repair_failed", str(e)[:200]))
                continue
            fields = {**d.fields, **recovered}
            still = missing_fields(t, fields)
            docs[t] = ExtractedDoc(t, d.source, fields, still,
                                   "rules+llm" if recovered else d.extraction_method)
            trace.append(_t("ExtractorAgent", "llm_repaired",
                            f"{t}: recovered {sorted(recovered) or 'nothing'}; "
                            f"still missing {still or 'none'}"))
        return {"docs": docs, "repair_attempts": state.get("repair_attempts", 0) + 1,
                "trace": trace, "path": ["llm_repair"]}

    def validate(state: CaseState):
        findings = validator.validate(state.get("docs", {}))
        return {"findings": findings, "path": ["validate"],
                "trace": [_t("ValidatorAgent", "validate", f"{len(findings)} finding(s): "
                             + ", ".join(sorted({f.code for f in findings})))]}

    def decide_node(state: CaseState):
        d = decide(state["findings"])
        return {"decision": d, "path": ["decide"], "trace": [_t("ReporterAgent", "decide", d)]}

    def investigate(state: CaseState):
        inv = investigator.investigate(state.get("docs", {}), state["findings"],
                                       state["decision"], state.get("lang"))
        tools_used = [s["tool"] for s in inv["steps"] if s.get("type") == "tool"]
        return {"investigation": inv, "path": ["investigate"],
                "trace": [_t("InvestigatorAgent", "investigate",
                             f"{inv['mode']}: {len(tools_used)} tool call(s) -> "
                             + " > ".join(tools_used))]}

    def report(state: CaseState):
        inv = state.get("investigation")
        if inv and inv.get("summary") and inv["mode"].startswith("llm"):
            summary = inv["summary"]
        else:
            summary = reporter.summarise(state["decision"], state["findings"])
        return {"summary": summary, "path": ["report"],
                "trace": [_t("ReporterAgent", "report", "summary written")]}

    # ------------------------------------------------------------------ routing
    def after_extract(state: CaseState) -> str:
        needs = any(d.missing for d in state.get("docs", {}).values())
        if needs and llm.available and state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
            return "llm_repair"
        return "validate"

    def after_decide(state: CaseState) -> str:
        return "report" if state["decision"] == "AUTO_PASS" else "investigate"

    # ------------------------------------------------------------------ wiring
    g = StateGraph(CaseState)
    for name, fn in [("ingest", ingest), ("classify", classify), ("extract", extract),
                     ("llm_repair", llm_repair), ("validate", validate), ("decide", decide_node),
                     ("investigate", investigate), ("report", report)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "classify")
    g.add_edge("classify", "extract")
    g.add_conditional_edges("extract", after_extract,
                            {"llm_repair": "llm_repair", "validate": "validate"})
    g.add_edge("llm_repair", "validate")
    g.add_edge("validate", "decide")
    g.add_conditional_edges("decide", after_decide,
                            {"report": "report", "investigate": "investigate"})
    g.add_edge("investigate", "report")
    g.add_edge("report", END)
    return g.compile()


def initial_state(folder: str | Path, lang: str | None = None) -> CaseState:
    folder = Path(folder)
    return {"case_id": folder.name, "folder": str(folder),
            "lang": lang or os.getenv("TRADEDOC_LANG", "English"),
            "repair_attempts": 0, "trace": [], "path": []}
