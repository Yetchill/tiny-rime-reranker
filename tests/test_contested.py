import json

from data_pipeline.validate_dataset import validate


def row(document, target, candidates):
    return {
        "context": document,
        "pinyin": ["shi", "shi"],
        "candidates": [{"text": text, "rank": index} for index, text in enumerate(candidates)],
        "target_index": candidates.index(target),
        "source_document_id": document,
    }


def test_contested_requires_different_gold_targets(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = [
        row("doc:1", "事实", ["事实", "实施"]),
        row("doc:2", "事实", ["事实", "实施"]),
    ]
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
    assert validate([path])["contested_pinyin_keys"] == 0
    rows[1] = row("doc:2", "实施", ["事实", "实施"])
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
    assert validate([path])["contested_pinyin_keys"] == 1
