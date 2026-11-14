"""CLI:  python -m tradedoccheck data/samples/case_02_discrepancies
        python -m tradedoccheck data/samples --all --llm gemini --lang Thai
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_dotenv
from .llm import LLMError, get_llm
from .pipeline import Pipeline


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Thai text on Windows cmd
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    ap = argparse.ArgumentParser(prog="tradedoccheck",
                                 description="Cross-check import/export trade documents.")
    ap.add_argument("path", help="case folder, or a folder of case folders with --all")
    ap.add_argument("--all", action="store_true", help="treat PATH as a folder of cases")
    ap.add_argument("-o", "--out", default="outputs", help="output directory")
    ap.add_argument("--llm", choices=["none", "gemini", "anthropic", "ollama"],
                    help="override TRADEDOC_LLM")
    ap.add_argument("--lang", help="language of agent summaries, e.g. Thai (default English)")
    args = ap.parse_args()

    try:
        llm = get_llm(args.llm)
    except LLMError as e:
        sys.exit(f"Error: {e}")
    pipe = Pipeline(llm, lang=args.lang)
    root = Path(args.path)
    cases = sorted(p for p in root.iterdir() if p.is_dir()) if args.all else [root]
    print(f"Engine: {pipe.engine} | LLM: {llm.name} {llm.model}")
    for case in cases:
        r = pipe.process_case(case)
        pipe.save(r, args.out)
        counts = {s: sum(f.severity == s for f in r.findings) for s in ("HIGH", "MEDIUM", "LOW")}
        print(f"{r.case_id:<30} {r.decision:<9} H={counts['HIGH']} M={counts['MEDIUM']} "
              f"L={counts['LOW']}  path: {' > '.join(r.path)}")
    print(f"Reports written to {args.out}/")


if __name__ == "__main__":
    main()
