"""Pluggable LLM back-ends: Gemini (default for the demo), Claude, Ollama, or none.

Two capabilities:
  * complete(system, prompt)          -> plain text (classification, extraction repair, summary)
  * tool-calling conversation         -> used by the InvestigatorAgent, where the LLM decides
                                         which tool to call next.

All back-ends talk to the REST APIs with urllib, so no vendor SDK is required.
The LLM never decides pass/fail -- that is done by the deterministic ValidatorAgent.

Environment variables (or a .env file):
  TRADEDOC_LLM   = none | gemini | anthropic | ollama
  TRADEDOC_MODEL = model id (defaults in config.DEFAULT_MODELS)
  GEMINI_API_KEY / ANTHROPIC_API_KEY / OLLAMA_HOST
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_MODELS


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema ("type": "object", ...)


@dataclass
class Conversation:
    """Back-end specific message history, kept opaque to the agent loop."""
    system: str
    tools: list[ToolSpec]
    messages: list[Any] = field(default_factory=list)


def _post(url: str, body: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        raise LLMError(f"HTTP {e.code}: {detail}") from e
    except Exception as e:  # network, timeout, bad JSON
        raise LLMError(str(e)) from e


class BaseLLM:
    """'No LLM' mode. Every agent has a rule-based path for this case."""
    name = "none"
    model = ""
    available = False

    def complete(self, system: str, prompt: str, json_mode: bool = False) -> str:
        raise LLMError("No LLM configured (rule-based mode)")

    # --- tool calling ---------------------------------------------------------------
    def start(self, system: str, prompt: str, tools: list[ToolSpec]) -> Conversation:
        raise LLMError("No LLM configured (rule-based mode)")

    def step(self, conv: Conversation) -> tuple[list[ToolCall], str]:
        """Ask the model for its next move. Returns (tool_calls, text)."""
        raise LLMError("No LLM configured (rule-based mode)")

    def add_results(self, conv: Conversation, results: list[tuple[ToolCall, Any]]) -> None:
        raise LLMError("No LLM configured (rule-based mode)")


# ======================================================================================
class GeminiLLM(BaseLLM):
    name = "gemini"
    available = True
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    def _call(self, body: dict) -> dict:
        data = _post(self.URL.format(model=self.model), body,
                     {"x-goog-api-key": self.api_key}, timeout=120)
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError(f"Gemini returned no candidates: {json.dumps(data)[:300]}")
        return cands[0].get("content") or {"role": "model", "parts": []}

    def complete(self, system: str, prompt: str, json_mode: bool = False) -> str:
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if json_mode:
            body["generationConfig"] = {"responseMimeType": "application/json"}
        content = self._call(body)
        return "".join(p.get("text", "") for p in content.get("parts", []))

    def start(self, system, prompt, tools):
        return Conversation(system, tools, [{"role": "user", "parts": [{"text": prompt}]}])

    def step(self, conv):
        body = {"systemInstruction": {"parts": [{"text": conv.system}]},
                "contents": conv.messages,
                "tools": [{"functionDeclarations": [
                    {"name": t.name, "description": t.description, "parameters": t.parameters}
                    for t in conv.tools]}]}
        content = self._call(body)
        content.setdefault("role", "model")
        conv.messages.append(content)  # keep verbatim (preserves thought signatures)
        calls, text = [], []
        for i, part in enumerate(content.get("parts", [])):
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append(ToolCall(fc.get("id") or f"call_{len(conv.messages)}_{i}",
                                      fc["name"], fc.get("args") or {}))
            elif part.get("text") and not part.get("thought"):
                text.append(part["text"])
        return calls, "".join(text)

    def add_results(self, conv, results):
        parts = []
        for call, result in results:
            resp = {"name": call.name, "response": {"result": result}}
            if not call.id.startswith("call_"):
                resp["id"] = call.id
            parts.append({"functionResponse": resp})
        conv.messages.append({"role": "user", "parts": parts})


# ======================================================================================
class AnthropicLLM(BaseLLM):
    name = "anthropic"
    available = True
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    def _call(self, body: dict) -> dict:
        return _post(self.URL, {"model": self.model, "max_tokens": 2000, **body},
                     {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, timeout=120)

    def complete(self, system, prompt, json_mode=False):
        data = self._call({"system": system, "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def start(self, system, prompt, tools):
        return Conversation(system, tools, [{"role": "user", "content": prompt}])

    def step(self, conv):
        data = self._call({"system": conv.system, "messages": conv.messages,
                           "tools": [{"name": t.name, "description": t.description,
                                      "input_schema": t.parameters} for t in conv.tools]})
        blocks = data.get("content", [])
        conv.messages.append({"role": "assistant", "content": blocks})
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {})
                 for b in blocks if b.get("type") == "tool_use"]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return calls, text

    def add_results(self, conv, results):
        conv.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id,
             "content": json.dumps(r, ensure_ascii=False, default=str)} for c, r in results]})


# ======================================================================================
class OllamaLLM(BaseLLM):
    name = "ollama"
    available = True

    def __init__(self, model: str, host: str):
        self.model, self.host = model, host.rstrip("/")

    def _call(self, body: dict) -> dict:
        return _post(f"{self.host}/api/chat", {"model": self.model, "stream": False, **body},
                     {}, timeout=300).get("message", {})

    def complete(self, system, prompt, json_mode=False):
        body = {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}]}
        if json_mode:
            body["format"] = "json"
        return self._call(body).get("content", "")

    def start(self, system, prompt, tools):
        return Conversation(system, tools, [{"role": "system", "content": system},
                                            {"role": "user", "content": prompt}])

    def step(self, conv):
        msg = self._call({"messages": conv.messages, "tools": [
            {"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.parameters}} for t in conv.tools]})
        conv.messages.append({"role": "assistant", "content": msg.get("content", ""),
                              "tool_calls": msg.get("tool_calls", [])})
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(f"call_{len(conv.messages)}_{i}", fn.get("name", ""), args))
        return calls, msg.get("content", "")

    def add_results(self, conv, results):
        for call, result in results:
            conv.messages.append({"role": "tool", "tool_name": call.name,
                                  "content": json.dumps(result, ensure_ascii=False, default=str)})


# ======================================================================================
def get_llm(kind: str | None = None, api_key: str | None = None,
            model: str | None = None) -> BaseLLM:
    kind = (kind or os.getenv("TRADEDOC_LLM", "none")).lower()
    model = model or os.getenv("TRADEDOC_MODEL") or DEFAULT_MODELS.get(kind, "")
    if kind == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError("Gemini selected but GEMINI_API_KEY is not set")
        return GeminiLLM(key, model)
    if kind == "anthropic":
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("Claude selected but ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(key, model)
    if kind == "ollama":
        return OllamaLLM(model, os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    return BaseLLM()


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (tolerates ```json fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("No JSON object in LLM reply")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from LLM: {e}") from e
