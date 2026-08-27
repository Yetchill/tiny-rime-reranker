import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark.offline.export_prediction_artifact import stable_example_id
from benchmark.offline.error_overlap import (
    PredictionArtifactError,
    analyze,
    deterministic_sample,
    grouped_accuracy,
    hybrid_metrics,
    oracle_metrics,
    overlap_matrix,
    render_markdown,
    target_bucket,
    tune_hybrid_threshold,
    validate_records,
    win_loss,
)


def record(example_id, split, gold, wanxiang, tiny, confidence=0.9, *, context="上下文", contested=False):
    return {
        "example_id": example_id,
        "split": split,
        "source_document_id": f"doc:{example_id}",
        "context": context,
        "pinyin": ["shi", "shi"],
        "gold": gold,
        "candidates": [
            {"text": "A", "rime_rank": 0},
            {"text": "B", "rime_rank": 1},
            {"text": "C", "rime_rank": 2},
        ],
        "ambiguity_count": 3,
        "contested": contested,
        "methods": {
            "wanxiang": {"text": wanxiang},
            "tiny_8m": {"text": tiny, "confidence": confidence, "margin": 0.4},
        },
    }


def quadrant_records():
    return [
        record("1", "test", "A", "A", "A"),
        record("2", "test", "B", "A", "B"),
        record("3", "test", "B", "B", "A"),
        record("4", "test", "B", "A", "A"),
    ]


class ErrorOverlapTests(unittest.TestCase):
    def test_cli_writes_summary_and_error_samples(self):
        records = [
            record("v1", "val", "B", "A", "B", 0.9),
            record("v2", "val", "A", "A", "B", 0.2),
            record("t1", "test", "B", "A", "B", 0.95),
            record("t2", "test", "A", "A", "B", 0.1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in records),
                encoding="utf-8",
            )
            output = root / "analysis"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmark.offline.error_overlap",
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(output),
                    "--methods",
                    "wanxiang,tiny_8m",
                ],
                check=True,
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["samples"]["test"], 2)
            self.assertTrue((output / "wanxiang-wrong-tiny-right.jsonl").exists())
            self.assertTrue((output / "summary.md").exists())

    def test_example_id_is_stable_and_split_scoped(self):
        first = stable_example_id("test", "doc:1", "上下文", ["shi", "shi"], "事实")
        second = stable_example_id("test", "doc:1", "上下文", ["shi", "shi"], "事实")
        validation = stable_example_id("val", "doc:1", "上下文", ["shi", "shi"], "事实")
        self.assertEqual(first, second)
        self.assertNotEqual(first, validation)

    def test_overlap_matrix_and_oracle_accuracy(self):
        records = quadrant_records()
        overlap = overlap_matrix(records, "wanxiang", "tiny_8m")
        self.assertEqual(overlap["both_correct"], 1)
        self.assertEqual(overlap["left_only_correct"], 1)
        self.assertEqual(overlap["right_only_correct"], 1)
        self.assertEqual(overlap["both_wrong"], 1)
        oracle = oracle_metrics(records, "wanxiang", "tiny_8m")
        self.assertEqual(oracle["left_accuracy"], 0.5)
        self.assertEqual(oracle["right_accuracy"], 0.5)
        self.assertEqual(oracle["oracle_accuracy"], 0.75)
        self.assertEqual(oracle["oracle_gain_over_left"], 0.25)

    def test_win_loss_counts_directional_changes(self):
        self.assertEqual(
            win_loss(quadrant_records(), "wanxiang", "tiny_8m"),
            {
                "baseline": "wanxiang",
                "method": "tiny_8m",
                "wins": 1,
                "losses": 1,
                "net_wins": 0,
            },
        )

    def test_grouping_keeps_empty_expected_subsets(self):
        groups = grouped_accuracy(
            [record("1", "test", "A", "A", "A")],
            ("wanxiang", "tiny_8m"),
            target_bucket,
            ("1 char", "2 char"),
        )
        self.assertEqual(groups["1 char"]["samples"], 1)
        self.assertEqual(groups["2 char"]["samples"], 0)
        self.assertIsNone(groups["2 char"]["accuracy"]["wanxiang"])

    def test_deterministic_sampling_is_order_independent(self):
        records = [record(str(index), "test", "A", "A", "A") for index in range(30)]
        first = deterministic_sample(records, 10, 42)
        second = deterministic_sample(list(reversed(records)), 10, 42)
        self.assertEqual(
            [item["example_id"] for item in first],
            [item["example_id"] for item in second],
        )

    def test_hybrid_threshold_uses_val_only_and_applies_once_to_test(self):
        val = [
            record("v1", "val", "B", "A", "B", 0.9),
            record("v2", "val", "A", "A", "B", 0.2),
        ]
        selected = tune_hybrid_threshold(val, "wanxiang", "tiny_8m")
        self.assertEqual(selected["status"], "AVAILABLE")
        self.assertEqual(selected["threshold"], 0.9)
        self.assertEqual(selected["top1"], 1.0)
        test = [record("t1", "test", "B", "A", "B", 0.95)]
        result = hybrid_metrics(test, "wanxiang", "tiny_8m", selected["threshold"])
        self.assertEqual(result["top1"], 1.0)
        self.assertEqual(result["wins"], 1)
        with self.assertRaisesRegex(PredictionArtifactError, "val records only"):
            tune_hybrid_threshold(test, "wanxiang", "tiny_8m")

    def test_missing_confidence_marks_hybrid_unavailable(self):
        value = record("v1", "val", "A", "A", "A")
        del value["methods"]["tiny_8m"]["confidence"]
        self.assertEqual(
            tune_hybrid_threshold([value], "wanxiang", "tiny_8m")["status"],
            "NOT AVAILABLE",
        )

    def test_missing_required_field_and_out_of_set_prediction_fail(self):
        value = record("1", "test", "A", "A", "A")
        del value["source_document_id"]
        with self.assertRaisesRegex(PredictionArtifactError, "source_document_id"):
            validate_records([value], ("wanxiang", "tiny_8m"))
        value = record("2", "test", "A", "A", "outside")
        with self.assertRaisesRegex(PredictionArtifactError, "outside candidate list"):
            validate_records([value], ("wanxiang", "tiny_8m"))

    def test_empty_subset_metrics_do_not_divide_by_zero(self):
        overlap = overlap_matrix([], "wanxiang", "tiny_8m")
        self.assertEqual(overlap["samples"], 0)
        self.assertIsNone(overlap["proportions"]["both_correct"])
        oracle = oracle_metrics([], "wanxiang", "tiny_8m")
        self.assertIsNone(oracle["oracle_accuracy"])
        rendered = render_markdown(analyze([], ("wanxiang", "tiny_8m"), 1))
        self.assertIn("oracle accuracy: N/A", rendered)

    def test_analyze_reports_contested_and_grouped_metrics(self):
        records = [
            record("v1", "val", "B", "A", "B", 0.9, contested=True),
            record("v2", "val", "A", "A", "B", 0.2),
            record("t1", "test", "B", "A", "B", 0.95, context="一二三四五", contested=True),
            record("t2", "test", "A", "A", "B", 0.1),
        ]
        result = analyze(records, ("wanxiang", "tiny_8m"), 7)
        self.assertEqual(result["samples"], {"val": 2, "test": 2, "contested_test": 1})
        self.assertEqual(
            result["contested"]["wanxiang_vs_tiny_8m"]["oracle"]["oracle_accuracy"],
            1.0,
        )
        self.assertEqual(result["groups"]["context_length"]["5-8"]["samples"], 1)
        self.assertEqual(result["simple_hybrid"]["tuning"]["selected_on"], "val")
        self.assertEqual(result["wanxiang_vs_tiny8_quadrants"]["tiny8_only_correct"], 1)


if __name__ == "__main__":
    unittest.main()
