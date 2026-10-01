"""CPU-only checks of nontrivial aggregation and historical scoring behavior."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reproduce_scores", ROOT / "scripts/reproduce_scores.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


class ScoreTests(unittest.TestCase):
    def test_fresh_inference_accepts_score_difference_but_not_missing_data(self):
        result = {"all_screenshot_rows_match": False,
                  "all_dataset_counts_valid_and_no_errors": True,
                  "all_stored_correct_flags_match": True,
                  "all_pope_and_mmk12_group_counts_valid": True}
        self.assertFalse(S.verification_passes(result))
        self.assertTrue(S.verification_passes(result, allow_different_scores=True))
        result["all_dataset_counts_valid_and_no_errors"] = False
        self.assertFalse(S.verification_passes(result, allow_different_scores=True))

    def test_pope_overlapping_categories_and_pooling(self):
        rows = [
            {"prediction": "Yes", "reference": "Yes", "metadata": {"category": "adversarial,popular"}},
            {"prediction": "No", "reference": "Yes", "metadata": {"category": "adversarial"}},
            {"prediction": "No", "reference": "No", "metadata": {"category": "random"}},
        ]
        result = S.pope_scores(rows)
        self.assertEqual(result["pooled"]["total"], 4)
        self.assertAlmostEqual(result["pooled"]["f1"], 0.8)
        self.assertAlmostEqual(result["by_category"]["adversarial"]["f1"], 2 / 3)

    def test_chartqa_vendor_numeric_and_text(self):
        matcher = S.get_chartqa_matcher()
        self.assertTrue(S.chartqa_hit({"prediction": "104", "reference": "100"}, matcher))
        self.assertFalse(S.chartqa_hit({"prediction": "106", "reference": "100"}, matcher))
        self.assertTrue(S.chartqa_hit({"prediction": "BLUE", "reference": "['red', 'blue']"}, matcher))
        self.assertTrue(S.chartqa_hit({"prediction": "0.5", "reference": "50%"}, matcher))
        # Preserve vendor treatment of numeric zero; do not silently improve it.
        self.assertFalse(S.chartqa_hit({"prediction": "0.0", "reference": "0"}, matcher))

    def test_hallusion_group_accuracy_not_item_accuracy(self):
        rows = [
            {"index": "x_x_x_1_1_1", "metadata": {}, "correct": True},
            {"index": "x_x_x_1_1_2", "metadata": {}, "correct": False},
            {"index": "x_x_x_1_2_1", "metadata": {}, "correct": True},
            {"index": "x_x_x_1_2_2", "metadata": {}, "correct": True},
        ]
        result = S.hallusion_scores(rows)
        self.assertEqual(result["aAcc"], 75)
        self.assertEqual(result["fAcc"], 50)
        self.assertEqual(result["qAcc"], 50)
        self.assertAlmostEqual(result["leaderboard_Avg"], 175 / 3)

    def test_original_fallback_explicit_and_detected(self):
        rows = [{"position": 0, "reference": "C", "prediction": "D", "correct": True, "original_correct": True}]
        fallback, audit = S.score_records(rows, "AI2D_TEST")
        self.assertTrue(fallback[0]["correct"])
        self.assertEqual(audit["original_correct_fallback_count"], 1)
        strict, audit = S.score_records(rows, "AI2D_TEST", preserve_original=False)
        self.assertFalse(strict[0]["correct"])
        self.assertEqual(audit["stored_flag_mismatch_count"], 1)

    def test_historical_blank_a_corner_case_is_audited(self):
        rows = [{"position": 0, "reference": "A", "prediction": "", "correct": True}]
        scored, audit = S.score_records(rows, "AI2D_TEST")
        self.assertTrue(scored[0]["correct"])
        self.assertEqual(audit["empty_predictions_accepted_by_historical_matcher"], [0])

    def test_missing_or_error_is_not_accepted(self):
        rows = [{"position": 0, "reference": "A", "prediction": "", "error": "failed", "correct": False}]
        scored, _ = S.score_records(rows, "AI2D_TEST")
        self.assertFalse(scored[0]["correct"])


if __name__ == "__main__":
    unittest.main()
