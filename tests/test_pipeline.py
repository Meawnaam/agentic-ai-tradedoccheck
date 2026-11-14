import json
from pathlib import Path

import pytest

from tradedoccheck.llm import BaseLLM
from tradedoccheck.pipeline import Pipeline

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
CASES = sorted(p for p in SAMPLES.iterdir() if p.is_dir())


@pytest.fixture(scope="module")
def pipe():
    return Pipeline(BaseLLM())  # force offline / rule-based mode


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_findings_match_ground_truth(pipe, case):
    expected = set(json.loads((case / "expected.json").read_text())["expected_codes"])
    result = pipe.process_case(case)
    got = {f.code for f in result.findings}
    assert got == expected, f"unexpected={got - expected} missing={expected - got}"


def test_clean_case_auto_passes(pipe):
    r = pipe.process_case(SAMPLES / "case_01_clean")
    assert r.decision == "AUTO_PASS" and not r.findings
    assert len(r.documents) == 3 and all(not d.missing for d in r.documents)


def test_serious_case_is_held(pipe):
    assert pipe.process_case(SAMPLES / "case_02_discrepancies").decision == "HOLD"


def test_extraction_values(pipe):
    r = pipe.process_case(SAMPLES / "case_01_clean")
    inv = next(d for d in r.documents if d.doc_type == "invoice").fields
    assert inv["invoice_no"] == "INV-2026-0417"
    assert inv["total_amount"] == 14600.0
    assert len(inv["items"]) == 3 and inv["items"][0]["qty"] == 1000
