from __future__ import annotations

import hashlib


def normalized_pinyin(syllables: list[str] | tuple[str, ...]) -> str:
    return "'".join(syllable.strip().lower() for syllable in syllables if syllable.strip())


def stable_example_id(split: str, source_document_id: str, context: str, pinyin: list[str], gold: str) -> str:
    value = "\0".join((split, source_document_id, context, normalized_pinyin(pinyin), gold))
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def stable_priority(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{value}".encode()).digest(), "big")
