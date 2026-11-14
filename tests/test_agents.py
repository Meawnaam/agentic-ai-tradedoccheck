"""Tests for the LLM-driven paths, using a scripted fake LLM (no network, no API key)."""
import json
from pathlib import Path

import pytest

import tradedoccheck.llm as llm_mod
from tradedoccheck.llm import BaseLLM, Conversation, LLMError, ToolCall, get_llm
from tradedoccheck.pipeline import Pipeline
from tradedoccheck.tools import TOOL_SPECS

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


class FakeLLM(BaseLLM):
    """Answers extraction-repair prompts and plays a fixed tool-calling script."""
    name, model, available = "fake", "scripted", True

    def __init__(self, script=None, fail=False):
        self.script = list(script or [])
        self.fail = fail
        self.tool_results = []

    def complete(self, system, prompt, json_mode=False):
        if self.fail:
            raise LLMError("simulated outage")
        if "Fields needed" in prompt:
            return json.dumps({"invoice_no": "INV-2026-0417", "buyer": "Pacific Fresh Foods Inc."})
        return "LLM summary."

    def start(self, system, prompt, tools):
        return Conversation(system, tools, [prompt])

    def step(self, conv):
        if self.fail:
            raise LLMError("simulated outage")
        return (self.script.pop(0) if self.script else []), "thinking..."

    def add_results(self, conv, results):
        self.tool_results.extend(results)


# --------------------------------------------------------------------------- graph routing
def test_rule_mode_path_skips_repair():
    r = Pipeline(BaseLLM()).process_case(SAMPLES / "case_05_unusual_labels")
    assert "llm_repair" not in r.path and r.decision == "REVIEW"


def test_llm_repair_branch_recovers_fields_and_passes():
    r = Pipeline(FakeLLM()).process_case(SAMPLES / "case_05_unusual_labels")
    assert r.path[:4] == ["ingest", "classify", "extract", "llm_repair"]
    inv = next(d for d in r.documents if d.doc_type == "invoice")
    assert inv.extraction_method == "rules+llm" and not inv.missing
    assert r.decision == "AUTO_PASS" and "investigate" not in r.path


def test_clean_case_never_investigates():
    r = Pipeline(FakeLLM()).process_case(SAMPLES / "case_01_clean")
    assert r.path == ["ingest", "classify", "extract", "validate", "decide", "report"]


# --------------------------------------------------------------------------- investigator
def test_llm_investigator_chooses_tools_and_submits():
    script = [
        [ToolCall("1", "list_findings", {"severity": "HIGH"})],
        [ToolCall("2", "compare_items", {"description": "Frozen Shrimp 16/20"}),
         ToolCall("3", "compare_quantities", {})],
        [ToolCall("4", "submit_report", {"summary": "สรุป: ต้องแก้เอกสาร",
                                         "actions": ["Ask seller"], "escalate": True})],
    ]
    fake = FakeLLM(script)
    r = Pipeline(fake).process_case(SAMPLES / "case_02_discrepancies")
    inv = r.investigation
    assert inv["mode"].startswith("llm")
    assert [s["tool"] for s in inv["steps"] if s["type"] == "tool"] == [
        "list_findings", "compare_items", "compare_quantities", "submit_report"]
    assert r.summary == "สรุป: ต้องแก้เอกสาร"   # report node uses the agent's summary
    assert r.decision == "HOLD"                  # agent cannot change the decision
    shrimp = dict((c.name, res) for c, res in fake.tool_results)["compare_items"]
    assert shrimp["qty_difference"] == -100 and shrimp["value_at_stake"] == 850.0


def test_llm_failure_falls_back_to_rules():
    r = Pipeline(FakeLLM(fail=True)).process_case(SAMPLES / "case_04_dates_and_parties")
    assert r.investigation["mode"].startswith("rules (LLM failed")
    assert r.investigation["email_to_shipper"] and r.decision == "HOLD"


def test_unknown_tool_is_reported_not_crashing():
    script = [[ToolCall("1", "delete_everything", {})], []]
    r = Pipeline(FakeLLM(script)).process_case(SAMPLES / "case_03_missing_bl")
    step = next(s for s in r.investigation["steps"] if s["type"] == "tool")
    assert "unknown tool" in step["result"]


# --------------------------------------------------------------------------- providers
def _capture(monkeypatch, reply):
    sent = {}

    def fake_post(url, body, headers, timeout):
        sent.update(url=url, body=body, headers=headers)
        return reply
    monkeypatch.setattr(llm_mod, "_post", fake_post)
    return sent


def test_gemini_tool_round_trip(monkeypatch):
    reply = {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "check_dates", "args": {}}, "thoughtSignature": "sig"}]}}]}
    sent = _capture(monkeypatch, reply)
    g = get_llm("gemini", api_key="test-key", model="gemini-3.8-flash")
    conv = g.start("sys", "hello", TOOL_SPECS)
    calls, _ = g.step(conv)
    assert sent["url"].endswith("/models/gemini-3.8-flash:generateContent")
    assert sent["headers"]["x-goog-api-key"] == "test-key"
    assert "temperature" not in json.dumps(sent["body"])
    assert calls[0].name == "check_dates"
    assert conv.messages[-1]["parts"][0]["thoughtSignature"] == "sig"  # kept verbatim
    g.add_results(conv, [(calls[0], {"ok": True})])
    assert conv.messages[-1]["parts"][0]["functionResponse"]["name"] == "check_dates"


def test_anthropic_tool_round_trip(monkeypatch):
    reply = {"content": [{"type": "tool_use", "id": "tu1", "name": "compare_parties",
                          "input": {}}]}
    _capture(monkeypatch, reply)
    a = get_llm("anthropic", api_key="k")
    conv = a.start("sys", "hi", TOOL_SPECS)
    calls, _ = a.step(conv)
    a.add_results(conv, [(calls[0], {"x": 1})])
    assert conv.messages[-1]["content"][0]["tool_use_id"] == "tu1"


def test_ollama_tool_round_trip(monkeypatch):
    reply = {"message": {"content": "", "tool_calls": [
        {"function": {"name": "lookup_hs_code", "arguments": {"code": "0306.17"}}}]}}
    _capture(monkeypatch, reply)
    o = get_llm("ollama")
    conv = o.start("sys", "hi", TOOL_SPECS)
    calls, _ = o.step(conv)
    assert calls[0].args == {"code": "0306.17"}


def test_missing_key_is_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        get_llm("gemini")


def test_dotenv_loader(tmp_path, monkeypatch):
    from tradedoccheck.config import load_dotenv
    monkeypatch.delenv("TDC_TEST_VAR", raising=False)
    (tmp_path / ".env").write_text('# comment\nTDC_TEST_VAR="abc"\n', encoding="utf-8")
    load_dotenv(tmp_path / ".env")
    import os
    assert os.environ["TDC_TEST_VAR"] == "abc"
