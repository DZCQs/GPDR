"""Reference-category weights and model checkpoint compatibility."""
import pickle

import numpy as np
import pytest
import torch

from gpdr import BetaBase, GPDR


def make_model(C):
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.float64).reshape(-1, 1)
    y = 0.2 + 0.4 * x[:, 0]
    points = torch.tensor([[0.2, 0.3], [0.5, 0.6], [0.8, 0.7]], dtype=x.dtype)
    return GPDR(x, y, base=BetaBase(0.5, 4.0), inducing_points=points, C=C)


@pytest.mark.parametrize('C', [1, 2, 3])
def test_initial_weights_and_random_stream(C):
    torch.manual_seed(23)
    initial_logits = torch.randn(C, dtype=torch.float64)
    torch.randn(C, 3, dtype=torch.float64)
    for _ in range(C):
        torch.randn(3, dtype=torch.float64)
    expected_rng = torch.get_rng_state()

    torch.manual_seed(23)
    model = make_model(C)
    assert model.logits.shape == (C - 1,)
    assert torch.equal(model.logits, initial_logits[:-1] - initial_logits[-1])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(model._weights(), torch.softmax(initial_logits, 0))
    loss, _ = model.forward_objective()
    loss.backward()
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()
    if C > 1:
        before = model._weights().detach().clone()
        with torch.no_grad():
            model.logits[0] += 0.5
        after = model._weights()
        assert after[-1] != before[-1]
        torch.testing.assert_close(after.sum(), after.new_tensor(1.0))


@pytest.mark.parametrize('C', [1, 2, 3])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_state_dict_preserves_distribution(C, legacy, nested):
    model = make_model(C)
    original = model._weights().detach().clone()
    source = torch.nn.ModuleDict({'model': model}) if nested else model
    checkpoint = source.state_dict()
    key = 'model.logits' if nested else 'logits'
    if legacy:
        checkpoint[key] = torch.cat((model.logits.detach(), model.logits.new_zeros(1))) + 1.25
    saved_logits = checkpoint[key].clone()
    restored = make_model(C)
    target = torch.nn.ModuleDict({'model': restored}) if nested else restored
    target.load_state_dict(checkpoint)
    assert torch.equal(checkpoint[key], saved_logits)
    assert restored.logits.shape == (C - 1,)
    torch.testing.assert_close(restored._weights(), original)
    np.testing.assert_allclose(model.predict_density([0.3], M=51).density,
                               restored.predict_density([0.3], M=51).density,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('C', [1, 2, 3])
def test_legacy_full_model_pickle(C):
    model = make_model(C)
    original = model._weights().detach().clone()
    original_density = model.predict_density([0.3], M=51).density
    full_logits = torch.cat((model.logits.detach(), model.logits.new_zeros(1))) + 1.25
    model.logits = torch.nn.Parameter(full_logits, requires_grad=False)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.logits.shape == (C - 1,)
    assert not restored.logits.requires_grad
    torch.testing.assert_close(restored._weights(), original)
    np.testing.assert_allclose(restored.predict_density([0.3], M=51).density,
                               original_density, rtol=1e-12, atol=1e-12)
