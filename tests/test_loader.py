"""Tests for CasebookLoader."""

import pytest
from pathlib import Path
from investor_casebook.data.loader import CasebookLoader

DATA_DIR = Path(__file__).parent.parent / "src" / "investor_casebook" / "data"


def test_load_sample_cases():
    loader = CasebookLoader(DATA_DIR)
    cases = loader.load_cases("sample_cases.jsonl")
    assert len(cases) == 5
    assert cases[0]["category"] == "Portfolio Construction"
    assert cases[0]["id"] == "PM-RISK-001"


def test_all_cases_have_required_fields():
    loader = CasebookLoader(DATA_DIR)
    cases = loader.load_cases("sample_cases.jsonl")
    for case in cases:
        assert "id" in case
        assert "category" in case
        assert "prompt" in case
        assert "golden_answer" in case


def test_case_categories():
    loader = CasebookLoader(DATA_DIR)
    cases = loader.load_cases("sample_cases.jsonl")
    categories = {c["category"] for c in cases}
    assert "Portfolio Construction" in categories
    assert "Risk Decomposition" in categories
    assert "Regulatory" in categories
    assert "Position Sizing" in categories
    assert "Hedge Effectiveness" in categories


def test_file_not_found():
    loader = CasebookLoader(DATA_DIR)
    with pytest.raises(FileNotFoundError):
        loader.load_cases("missing.jsonl")
