import unittest

from benchmark.offline.candidate_union import bounded_union


class CandidateUnionTests(unittest.TestCase):
    def test_round_robin_union_is_deterministic_bounded_and_deduplicated(self):
        rime = [{"text": value} for value in ("A", "B", "C", "D")]
        wanxiang = [{"text": value} for value in ("B", "E", "A", "F")]
        result = bounded_union(rime, wanxiang, 5)
        self.assertEqual([value["text"] for value in result], ["A", "B", "E", "C", "D"])
        self.assertEqual(result[1], {"text": "B", "rime_rank": 1, "wanxiang_rank": 0})
        self.assertEqual(len({value["text"] for value in result}), 5)
        self.assertEqual(result, bounded_union(rime, wanxiang, 5))

    def test_union_handles_exhausted_source_and_rejects_zero_budget(self):
        self.assertEqual(
            [value["text"] for value in bounded_union([{"text": "A"}], [{"text": "B"}, {"text": "C"}], 4)],
            ["A", "B", "C"],
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            bounded_union([], [], 0)


if __name__ == "__main__":
    unittest.main()
