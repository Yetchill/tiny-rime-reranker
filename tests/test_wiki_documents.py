import json

from data_pipeline.wiki.sample_documents import jsonl_documents


def test_hf_jsonl_adapter_preserves_document_identity(tmp_path):
    source = tmp_path / "fixture.json"
    source.write_text(
        json.dumps({"id": 13, "title": "数学", "tags": ["形式科学"], "text": "数学是研究数量的学科。"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    [document] = jsonl_documents(source, "20260501")
    assert document["source_document_id"] == "zhwiki:20260501:13"
    assert document["title"] == "数学"
    assert document["tags"] == ["形式科学"]
