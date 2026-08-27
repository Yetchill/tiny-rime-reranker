import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.offline.freeze_benchmark import canonical_jsonl_hash, ensure_new_manifest


class BenchmarkManifestTests(unittest.TestCase):
    def test_canonical_hash_is_key_order_independent_and_tracks_document_set(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            a = {"source_document_id": "doc:2", "value": 1}
            b = {"value": 2, "source_document_id": "doc:1"}
            first.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
            second.write_text(json.dumps({"value": 1, "source_document_id": "doc:2"}) + "\n" + json.dumps(b) + "\n")
            first_hash, count, documents_hash = canonical_jsonl_hash(first)
            second_hash, _, _ = canonical_jsonl_hash(second)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(count, 2)
            self.assertEqual(documents_hash, hashlib.sha256("doc:1\ndoc:2".encode()).hexdigest())

    def test_frozen_manifest_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark_manifest.json"
            ensure_new_manifest(path)
            path.write_text("{}")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                ensure_new_manifest(path)


if __name__ == "__main__":
    unittest.main()
