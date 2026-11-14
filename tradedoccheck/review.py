"""Human-in-the-loop: record the officer's final decision for each case (JSON Lines file)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ACTIONS = ("APPROVE", "REJECT", "REQUEST_CORRECTION")


def save_review(out_dir: str | Path, case_id: str, system_decision: str, action: str,
                reviewer: str, comment: str = "") -> dict:
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}")
    rec = {"timestamp": datetime.now().isoformat(timespec="seconds"), "case_id": case_id,
           "system_decision": system_decision, "officer_action": action,
           "reviewer": reviewer or "anonymous", "comment": comment,
           "override": (system_decision == "HOLD" and action == "APPROVE")
                       or (system_decision == "AUTO_PASS" and action != "APPROVE")}
    path = Path(out_dir) / "reviews.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_reviews(out_dir: str | Path) -> list[dict]:
    path = Path(out_dir) / "reviews.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
