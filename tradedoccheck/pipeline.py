"""Public entry point: runs the LangGraph workflow over one shipment folder (a "case")."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .agents.reporter import ReporterAgent
from .graph import ENGINE, build_graph, initial_state
from .llm import BaseLLM, get_llm
from .models import CaseResult


class Pipeline:
    def __init__(self, llm: BaseLLM | None = None, lang: str | None = None):
        self.llm = llm or get_llm()
        self.lang = lang
        self.graph = build_graph(self.llm)
        self.engine = ENGINE

    def _result(self, state: dict) -> CaseResult:
        return CaseResult(
            case_id=state["case_id"], documents=list(state.get("docs", {}).values()),
            findings=state.get("findings", []), decision=state.get("decision", "REVIEW"),
            summary=state.get("summary", ""), trace=state.get("trace", []),
            investigation=state.get("investigation"), path=state.get("path", []),
            engine=self.engine,
            llm=f"{self.llm.name}:{self.llm.model}" if self.llm.available else "none")

    def process_case(self, folder: str | Path) -> CaseResult:
        return self._result(self.graph.invoke(initial_state(folder, self.lang)))

    def stream_case(self, folder: str | Path) -> Iterator[tuple[str, dict]]:
        """Yield (node_name, state_so_far) after every node, then ("__result__", CaseResult)."""
        state: dict = {}
        for state in self.graph.stream(initial_state(folder, self.lang), stream_mode="values"):
            if state.get("path"):
                yield state["path"][-1], state
        yield "__result__", self._result(state)

    @staticmethod
    def save(result: CaseResult, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{result.case_id}.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        (out / f"{result.case_id}.md").write_text(
            ReporterAgent.to_markdown(result), encoding="utf-8")
