from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    text: str
    rank: int
    quality: float | None = None
    type: str | None = None
    comment: str = ""
    preedit: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Candidate":
        candidate = cls(
            text=str(value["text"]),
            rank=int(value["rank"]),
            quality=(None if value.get("quality") is None else float(value["quality"])),
            type=(None if value.get("type") is None else str(value["type"])),
            comment=str(value.get("comment", "")),
            preedit=str(value.get("preedit", "")),
        )
        if not candidate.text:
            raise ValueError("candidate text must not be empty")
        if candidate.rank < 0:
            raise ValueError("candidate rank must be non-negative")
        return candidate


@dataclass(frozen=True)
class RankingExample:
    context: str
    pinyin: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    target_index: int
    source_document_id: str

    @classmethod
    def from_dict(cls, value: dict[str, Any], top_k: int = 8) -> "RankingExample":
        candidates = tuple(Candidate.from_dict(item) for item in value["candidates"])
        example = cls(
            context=str(value.get("context", ""))[-32:],
            pinyin=tuple(str(item) for item in value["pinyin"][:16]),
            candidates=candidates,
            target_index=int(value["target_index"]),
            source_document_id=str(value["source_document_id"]),
        )
        example.validate(top_k=top_k)
        return example

    def validate(self, top_k: int = 8) -> None:
        if not self.source_document_id:
            raise ValueError("source_document_id is required")
        if not self.pinyin:
            raise ValueError("at least one pinyin syllable is required")
        if not 1 <= len(self.candidates) <= top_k:
            raise ValueError(f"candidate count must be in [1, {top_k}]")
        if tuple(candidate.rank for candidate in self.candidates) != tuple(range(len(self.candidates))):
            raise ValueError("candidate ranks must be contiguous and preserve Rime order")
        if not 0 <= self.target_index < len(self.candidates):
            raise ValueError("target_index is out of range")
        if len({candidate.text for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate texts must be unique")
