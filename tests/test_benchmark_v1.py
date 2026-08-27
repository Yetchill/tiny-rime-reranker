import unittest

from data_pipeline.benchmark_v1 import (
    StableRecordSample,
    apply_contested_labels,
    canonical_ranking_example,
    deterministic_train_sample,
    miss_diagnostic,
)
from tinyrime.schema import RankingExample


def query(document, target, split=None):
    value = {
        "source_document_id": document,
        "context": f"context-{document}",
        "pinyin": ["shi", "shi"],
        "target": target,
        "proper_noun_proxy": False,
    }
    if split:
        value["split"] = split
    return value


def scored(example_id, target_index, contested=True):
    return {
        "example_id": example_id,
        "source_document_id": f"doc:{example_id}",
        "context": "上下文",
        "pinyin": ["shi", "shi"],
        "target": "实施",
        "candidates": [
            {"text": "事实", "rank": 0},
            {"text": "实施", "rank": 1},
            {"text": "时事", "rank": 2},
        ],
        "target_index": target_index,
        "contested": contested,
    }


class BenchmarkV1Tests(unittest.TestCase):
    def test_contested_is_canonical_across_full_query_pool(self):
        queries = {
            "train": [query("train:1", "事实")],
            "val": [query("val:1", "事实")],
            "test": [query("test:1", "实施")],
        }
        keys = apply_contested_labels(queries)
        self.assertEqual(keys, {"shi'shi"})
        self.assertTrue(queries["train"][0]["contested"])
        self.assertTrue(queries["val"][0]["contested"])
        self.assertTrue(queries["test"][0]["contested"])
        self.assertEqual(queries["test"][0]["split"], "test")
        self.assertEqual(len(queries["test"][0]["example_id"]), 24)

    def test_ranking_example_reads_persisted_contested_flag(self):
        raw = {
            "source_document_id": "doc:1",
            "context": "上下文",
            "pinyin": ["shi", "shi"],
            "candidates": [{"text": "事实", "rank": 0}, {"text": "实施", "rank": 1}],
            "target_index": 1,
            "contested": True,
        }
        self.assertTrue(RankingExample.from_dict(raw).contested)

    def test_canonical_ranking_example_filters_by_pool_without_relabeling(self):
        self.assertIsNone(canonical_ranking_example(scored("a", 8), 8))
        value = canonical_ranking_example(scored("b", 1, contested=True), 2)
        self.assertEqual(len(value["candidates"]), 2)
        self.assertEqual(value["target_index"], 1)
        self.assertTrue(value["contested"])
        missing = scored("c", 1)
        del missing["contested"]
        with self.assertRaisesRegex(ValueError, "canonical contested"):
            canonical_ranking_example(missing, 8)

    def test_train_sampling_is_stable_and_order_independent(self):
        values = [{"example_id": f"id:{index}"} for index in range(30)]
        first = deterministic_train_sample(values, 10, 20260827)
        second = deterministic_train_sample(list(reversed(values)), 10, 20260827)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        with self.assertRaisesRegex(ValueError, "requested 31"):
            deterministic_train_sample(values, 31, 1)

    def test_split_specific_miss_sampling_is_deterministic(self):
        values = [{"example_id": f"id:{index}", "split": "test"} for index in range(30)]
        first = StableRecordSample(7, 10)
        second = StableRecordSample(7, 10)
        for value in values:
            first.add(value)
        for value in reversed(values):
            second.add(value)
        self.assertEqual(first.records(), second.records())
        self.assertEqual(len(first.records()), 7)

    def test_miss_diagnostics_do_not_claim_unverified_oov(self):
        value = {
            **query("doc:1", "实施", split="test"),
            "example_id": "example",
            "contested": True,
        }
        empty = miss_diagnostic(value, [], -1, 8)
        self.assertEqual(empty["category"], "decoder_empty")
        absent = miss_diagnostic(value, [{"text": "事实"}], -1, 8)
        self.assertEqual(absent["category"], "absent_top_pool_oov_unknown")
        self.assertEqual(absent["oov_status"], "UNKNOWN_WITHOUT_DICTIONARY_LOOKUP")
        below = miss_diagnostic(value, [{"text": str(index)} for index in range(12)], 10, 8)
        self.assertEqual(below["category"], "below_top8")
        self.assertEqual(below["oov_status"], "NOT_OOV")


if __name__ == "__main__":
    unittest.main()
