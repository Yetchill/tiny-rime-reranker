from time import monotonic_ns

from tinyrime import Candidate, GateConfig, conservative_rerank


def candidates():
    return [Candidate("事实", 0), Candidate("实施", 1), Candidate("试试", 2)]


def test_abstains_without_context():
    result = conservative_rerank(candidates(), [0, 10, 0], 0.99, "")
    assert not result.changed
    assert result.reason == "no-context"
    assert [candidate.text for candidate in result.candidates] == ["事实", "实施", "试试"]


def test_promotes_only_existing_candidate():
    result = conservative_rerank(
        candidates(),
        [0, 8, 0],
        0.99,
        "项目正在",
        GateConfig(alpha=0.25, margin_threshold=0.1),
    )
    assert result.changed
    assert [candidate.text for candidate in result.candidates] == ["实施", "事实", "试试"]
    assert {candidate.text for candidate in result.candidates} == {"事实", "实施", "试试"}


def test_deadline_falls_back_to_original_order():
    result = conservative_rerank(
        candidates(),
        [0, 8, 0],
        0.99,
        "项目正在",
        start_ns=monotonic_ns() - 10_000_000,
    )
    assert not result.changed
    assert result.reason == "deadline"
