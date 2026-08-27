import pytest

from tinyrime import RankingExample


def sample():
    return {
        "context": "根据调查得到的",
        "pinyin": ["shi", "shi"],
        "candidates": [
            {"text": "事实", "rank": 0},
            {"text": "实施", "rank": 1},
        ],
        "target_index": 0,
        "source_document_id": "fixture:001",
    }


def test_valid_example():
    value = RankingExample.from_dict(sample())
    assert value.context == "根据调查得到的"


def test_rejects_non_contiguous_rank():
    value = sample()
    value["candidates"][1]["rank"] = 3
    with pytest.raises(ValueError, match="contiguous"):
        RankingExample.from_dict(value)
