from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns
from typing import Sequence

from .schema import Candidate


@dataclass(frozen=True)
class GateConfig:
    alpha: float = 0.25
    confidence_threshold: float = 0.80
    margin_threshold: float = 0.15
    deadline_ms: float = 3.0
    require_context: bool = True
    require_competition: bool = True


@dataclass(frozen=True)
class RerankDecision:
    candidates: tuple[Candidate, ...]
    changed: bool
    reason: str
    elapsed_ms: float


def conservative_rerank(
    candidates: Sequence[Candidate],
    residual_scores: Sequence[float],
    confidence: float,
    context: str,
    config: GateConfig = GateConfig(),
    *,
    start_ns: int | None = None,
) -> RerankDecision:
    """Return a stable permutation or the untouched Rime ordering.

    Backends call this once, before the candidate list is exposed. No late result
    can mutate a previously returned decision.
    """

    started = monotonic_ns() if start_ns is None else start_ns
    original = tuple(candidates)

    def abstain(reason: str) -> RerankDecision:
        return RerankDecision(original, False, reason, (monotonic_ns() - started) / 1e6)

    if not original:
        return abstain("empty")
    if len(original) != len(residual_scores):
        return abstain("invalid-score-count")
    if config.require_context and not context.strip():
        return abstain("no-context")
    if config.require_competition and len(original) < 2:
        return abstain("no-competition")
    if confidence < config.confidence_threshold:
        return abstain("low-confidence")
    if (monotonic_ns() - started) / 1e6 > config.deadline_ms:
        return abstain("deadline")

    base = [-(candidate.rank) for candidate in original]
    final = [base_score + config.alpha * float(residual) for base_score, residual in zip(base, residual_scores)]
    order = sorted(range(len(original)), key=lambda index: (-final[index], index))
    if order[0] == 0:
        return abstain("top1-unchanged")
    margin = final[order[0]] - final[0]
    if margin < config.margin_threshold:
        return abstain("low-margin")
    if (monotonic_ns() - started) / 1e6 > config.deadline_ms:
        return abstain("deadline")
    return RerankDecision(
        tuple(original[index] for index in order),
        True,
        "promoted",
        (monotonic_ns() - started) / 1e6,
    )
