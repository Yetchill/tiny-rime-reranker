import unittest
from collections import Counter

from training.candidate_types import CANDIDATE_TYPE_TO_ID, candidate_type_id
from training.vocabulary import ExactVocabulary, build_exact_vocabulary, collision_report, hash_token_id


class VocabularyTests(unittest.TestCase):
    def test_candidate_types_are_fixed_categorical_ids(self):
        self.assertEqual(candidate_type_id("table"), CANDIDATE_TYPE_TO_ID["table"])
        self.assertEqual(candidate_type_id(None), CANDIDATE_TYPE_TO_ID["<UNK>"])
        self.assertEqual(candidate_type_id("future_type"), CANDIDATE_TYPE_TO_ID["<UNK>"])
        self.assertNotEqual(candidate_type_id("table"), candidate_type_id("phrase"))

    def test_exact_vocabulary_is_deterministic_and_namespace_disjoint(self):
        characters = Counter({"事": 10, "实": 8, "shi": 1})
        pinyin = Counter({"shi": 20, "shi4": 2})
        first = build_exact_vocabulary(characters, pinyin, 16)
        second = build_exact_vocabulary(characters, pinyin, 16)
        self.assertEqual(first, second)
        self.assertEqual(first.character_id("missing"), 1)
        self.assertEqual(first.pinyin_id("missing"), 1)
        self.assertNotEqual(first.character_id("shi"), first.pinyin_id("shi"))
        self.assertEqual(first.required_embeddings, 7)

    def test_exact_vocabulary_round_trip_and_capacity_guard(self):
        vocabulary = build_exact_vocabulary(Counter({"甲": 1}), Counter({"jia": 1}), 8)
        self.assertEqual(ExactVocabulary.from_dict(vocabulary.to_dict()), vocabulary)
        with self.assertRaisesRegex(ValueError, "requires 4 embeddings"):
            build_exact_vocabulary(Counter({"甲": 1}), Counter({"jia": 1}), 3)

    def test_hash_collision_audit_reports_known_collision(self):
        characters = Counter({"甲": 10, "乙": 5})
        pinyin = Counter({"jia": 7})
        report = collision_report(characters, pinyin, capacity=2)
        self.assertEqual(report["unique_tokens"], 3)
        self.assertEqual(report["occupied_buckets"], 1)
        self.assertEqual(report["collision_count"], 2)
        self.assertEqual(report["collision_rate"], 2 / 3)
        self.assertGreater(report["top100_tokens_with_collision"], 0)

    def test_hash_id_reserves_pad_bucket(self):
        for token in ("甲", "shi", "punct"):
            self.assertGreaterEqual(hash_token_id(token, 32, "char:"), 1)
            self.assertLess(hash_token_id(token, 32, "char:"), 32)


if __name__ == "__main__":
    unittest.main()
