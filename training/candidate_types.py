from __future__ import annotations


CANDIDATE_TYPES = (
    "<PAD>",
    "<UNK>",
    "table",
    "user_table",
    "phrase",
    "user_phrase",
    "sentence",
    "completion",
    "predict",
    "history",
    "punct",
    "simplified",
)
CANDIDATE_TYPE_TO_ID = {value: index for index, value in enumerate(CANDIDATE_TYPES)}


def candidate_type_id(value: str | None) -> int:
    if not value:
        return CANDIDATE_TYPE_TO_ID["<UNK>"]
    return CANDIDATE_TYPE_TO_ID.get(value, CANDIDATE_TYPE_TO_ID["<UNK>"])
