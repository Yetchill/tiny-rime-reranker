from __future__ import annotations

import heapq
from collections import defaultdict

from data_pipeline.identity import normalized_pinyin, stable_example_id, stable_priority


def apply_contested_labels(queries_by_split: dict[str, list[dict]]) -> set[str]:
    targets_by_pinyin: dict[str, set[str]] = defaultdict(set)
    for queries in queries_by_split.values():
        for query in queries:
            targets_by_pinyin[normalized_pinyin(query["pinyin"])].add(query["target"])
    contested_keys = {key for key, targets in targets_by_pinyin.items() if len(targets) >= 2}
    for split, queries in queries_by_split.items():
        for query in queries:
            query["split"] = split
            query["example_id"] = stable_example_id(
                split,
                query["source_document_id"],
                query["context"],
                query["pinyin"],
                query["target"],
            )
            query["contested"] = normalized_pinyin(query["pinyin"]) in contested_keys
    return contested_keys


def canonical_ranking_example(scored: dict, pool_k: int) -> dict | None:
    target_index = int(scored["target_index"])
    if target_index < 0 or target_index >= pool_k:
        return None
    if "contested" not in scored or not isinstance(scored["contested"], bool):
        raise ValueError("scored query is missing canonical contested flag")
    return {
        "example_id": scored["example_id"],
        "source_document_id": scored["source_document_id"],
        "context": scored["context"],
        "pinyin": scored["pinyin"],
        "candidates": scored["candidates"][:pool_k],
        "target_index": target_index,
        "contested": scored["contested"],
    }


def deterministic_train_sample(examples: list[dict], maximum: int, seed: int) -> list[dict]:
    if len(examples) < maximum:
        raise ValueError(f"only {len(examples)} recallable train examples; requested {maximum}")
    return sorted(examples, key=lambda item: (stable_priority(seed, item["example_id"]), item["example_id"]))[
        :maximum
    ]


class StableRecordSample:
    def __init__(self, maximum: int, seed: int):
        self.maximum = maximum
        self.seed = seed
        self.heap: list[tuple[int, str, dict]] = []

    def add(self, record: dict) -> None:
        if self.maximum <= 0:
            return
        priority = stable_priority(self.seed, record["example_id"])
        item = (-priority, record["example_id"], record)
        if len(self.heap) < self.maximum:
            heapq.heappush(self.heap, item)
        elif priority < -self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def records(self) -> list[dict]:
        selected = [(-negative, example_id, record) for negative, example_id, record in self.heap]
        return [record for _, _, record in sorted(selected)]


def length_category(value: str) -> str:
    return str(len(value)) if len(value) <= 4 else "5+"


def miss_diagnostic(query: dict, candidates: list[dict], target_index: int, display_k: int) -> dict:
    if not candidates:
        category = "decoder_empty"
    elif target_index < 0:
        category = "absent_top_pool_oov_unknown"
    elif target_index >= display_k:
        category = f"below_top{display_k}"
    else:
        raise ValueError("diagnostic requested for a non-miss")
    return {
        "example_id": query["example_id"],
        "split": query["split"],
        "source_document_id": query["source_document_id"],
        "context": query["context"],
        "pinyin": query["pinyin"],
        "gold": query["target"],
        "contested": query["contested"],
        "target_index": target_index,
        "category": category,
        "oov_status": "UNKNOWN_WITHOUT_DICTIONARY_LOOKUP" if target_index < 0 else "NOT_OOV",
        "normalization_status": "passed_query_generation",
        "proper_noun_proxy": bool(query.get("proper_noun_proxy", False)),
        "target_length_category": length_category(query["target"]),
        "pinyin_length": len(query["pinyin"]),
        "candidate_count": len(candidates),
    }
