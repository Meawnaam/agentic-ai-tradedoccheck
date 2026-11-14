"""ClassifierAgent: decides what kind of trade document a PDF is."""
from __future__ import annotations

from ..llm import BaseLLM, LLMError
from ..models import DOC_TYPES

KEYWORDS = [  # order matters: most specific first
    ("bill_of_lading", ("bill of lading", "b/l no")),
    ("packing_list", ("packing list",)),
    ("invoice", ("commercial invoice", "invoice")),
]


class ClassifierAgent:
    name = "ClassifierAgent"

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def classify(self, text: str) -> tuple[str, str]:
        """Return (doc_type, method). doc_type is 'unknown' if it cannot be determined."""
        head = text[:600].lower()
        for doc_type, words in KEYWORDS:
            if any(w in head for w in words):
                return doc_type, "rules"
        if self.llm.available:
            try:
                reply = self.llm.complete(
                    "You classify international trade documents. Answer with exactly one of: "
                    + ", ".join(DOC_TYPES) + ", unknown.",
                    text[:1500])
                answer = reply.strip().lower().split()[0].strip(".,\"'")
                if answer in DOC_TYPES:
                    return answer, "llm"
            except (LLMError, IndexError):
                pass
        return "unknown", "rules"
