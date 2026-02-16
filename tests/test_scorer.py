"""Tests for CaseScorer — rule-based golden-answer comparison."""

from investor_casebook.reasoning.scorer import CaseScorer, _extract_numbers


class TestExtractNumbers:
    def test_dollar_amounts(self):
        nums = _extract_numbers("Tech Sleeve: -$21M, Treasury: -$10.5M")
        assert 21.0 in nums or -21.0 in nums
        assert 10.5 in nums or -10.5 in nums

    def test_percentages(self):
        nums = _extract_numbers("dropped 3% and beta is 1.4")
        assert 3.0 in nums
        assert 1.4 in nums

    def test_basis_points(self):
        nums = _extract_numbers("spread widened 50bps to 370bps")
        assert 50.0 in nums
        assert 370.0 in nums

    def test_mixed_financial_text(self):
        text = "P&L: -$21.1M. Beta 1.4, duration 7, yield +0.50%"
        nums = _extract_numbers(text)
        assert len(nums) >= 3  # Should find several numbers

    def test_empty_string(self):
        assert _extract_numbers("") == []

    def test_no_numbers(self):
        assert _extract_numbers("This is a qualitative assessment.") == []


class TestCaseScorer:
    def setup_method(self):
        self.scorer = CaseScorer()

    def test_perfect_match(self):
        golden = "Tech Sleeve: -$21M. Beta 1.4. Duration 7."
        model = "Tech Sleeve: -$21M. Beta 1.4. Duration 7."
        scores = self.scorer.score_case(model, golden)
        assert scores["completeness"] >= 0.9
        assert scores["numerical_accuracy"] >= 0.9
        assert scores["overall"] >= 0.9

    def test_no_overlap(self):
        golden = "Tech Sleeve: -$21M. Beta 1.4. Total P&L: -$21.1M."
        model = "I don't know the answer to this question."
        scores = self.scorer.score_case(model, golden)
        assert scores["completeness"] <= 0.2
        assert scores["overall"] <= 0.3

    def test_partial_match(self):
        golden = "Loss: -$21M. Beta 1.4. Hedge gain: +$10.4M."
        model = "The loss was approximately $21M due to beta exposure of 1.4."
        scores = self.scorer.score_case(model, golden)
        assert 0.3 < scores["completeness"] < 1.0
        assert scores["overall"] > 0.2

    def test_score_case_returns_all_keys(self):
        scores = self.scorer.score_case("output", "golden")
        assert "completeness" in scores
        assert "numerical_accuracy" in scores
        assert "structure" in scores
        assert "overall" in scores

    def test_score_all_aggregation(self):
        results = [
            {
                "id": "TEST-001",
                "category": "Test",
                "model_output": "Loss: -$21M. Beta 1.4.",
                "golden_answer": "Loss: -$21M. Beta 1.4.",
            },
            {
                "id": "TEST-002",
                "category": "Test",
                "model_output": "No relevant content here.",
                "golden_answer": "Gain: +$50M. Alpha: 2.5%.",
            },
        ]
        report = self.scorer.score_all(results)

        assert "per_case" in report
        assert "aggregate" in report
        assert len(report["per_case"]) == 2
        assert report["aggregate"]["count"] == 2
        assert report["per_case"][0]["id"] == "TEST-001"
        assert 0 <= report["aggregate"]["mean"] <= 1

    def test_score_all_empty(self):
        report = self.scorer.score_all([])
        assert report["per_case"] == []
        assert report["aggregate"] == {}
