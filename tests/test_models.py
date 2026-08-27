import pytest

torch = pytest.importorskip("torch")

from training.models import PRESETS, TinyContextReranker


@pytest.mark.parametrize("name", list(PRESETS))
def test_model_shapes_and_parameter_budget(name):
    config = PRESETS[name]
    model = TinyContextReranker(config)
    residuals, gate = model(
        torch.ones((2, 32), dtype=torch.long),
        torch.ones((2, 16), dtype=torch.long),
        torch.ones((2, 8, 8), dtype=torch.long),
        torch.zeros((2, 8, 4)),
    )
    assert residuals.shape == (2, 8)
    assert gate.shape == (2,)
    assert model.parameter_count < 10_000_000
