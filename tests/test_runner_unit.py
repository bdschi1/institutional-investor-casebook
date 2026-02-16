"""Unit tests for CasebookRunner — uses mock mode (no GPU required)."""

from investor_casebook.runner import CasebookRunner


def test_mock_init():
    """CasebookRunner initializes in mock mode without GPU."""
    runner = CasebookRunner(mock=True)
    assert runner.mock is True
    assert runner.model is None
    assert runner.tokenizer is None


def test_mock_run_case():
    """run_case returns a string in mock mode."""
    runner = CasebookRunner(mock=True)
    case = {"id": "TEST-001", "prompt": "Analyze this portfolio."}
    result = runner.run_case(case)
    assert isinstance(result, str)
    assert "TEST-001" in result
    assert "[MOCK]" in result


def test_mock_run_all_cases():
    """run_all_cases returns correct structure in mock mode."""
    runner = CasebookRunner(mock=True)
    cases = [
        {"id": "T-001", "category": "Test", "prompt": "Q1", "golden_answer": "A1"},
        {"id": "T-002", "category": "Test", "prompt": "Q2", "golden_answer": "A2"},
    ]
    results = runner.run_all_cases(cases)

    assert len(results) == 2
    assert results[0]["id"] == "T-001"
    assert results[1]["id"] == "T-002"
    assert "model_output" in results[0]
    assert "golden_answer" in results[0]
    assert isinstance(results[0]["model_output"], str)


def test_system_prompt_exists():
    """CasebookRunner has a system prompt."""
    assert hasattr(CasebookRunner, "SYSTEM_PROMPT")
    assert "Portfolio Manager" in CasebookRunner.SYSTEM_PROMPT
