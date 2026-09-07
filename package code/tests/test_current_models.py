import copy
import math
import pickle
import numpy as np
import pytest
import torch

from gpdr.base import StudentTLinearBase
from gpdr.kernels import KernelParams
from gpdr.models import LogisticGPMixtureDiag, BetaGPMixtureDiag
from gpdr.training import fit_adam
from gpdr.examples import prepare_toy_data, build_toy_model


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.mark.parametrize("C", [1, 2, 3])
@pytest.mark.parametrize("kind", ["student", "beta"])
def test_trainable_weights_and_stochastic_gradient(C, kind):
    torch.manual_seed(3)
    x = torch.linspace(0.05, 0.95, 24).reshape(-1, 1)
    y = 0.2 + x[:, 0] / 2
    kp = KernelParams(math.log(0.4**2), math.log(0.3**2), math.log(0.4**2))
    if kind == "student":
        model = LogisticGPMixtureDiag(x, y, base=StudentTLinearBase(1, 0.4), C=C, m_induce=9, kp=kp)
    else:
        model = BetaGPMixtureDiag(x, y, mu_base_train=np.full(24,0.5), phi_base=5.0, C=C, m_induce=9, kp=kp)
    loss, info = model.forward_objective()
    loss.backward()
    assert model.logits.shape == (C - 1,)
    assert info['w'].shape == (C,)
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert info['w'].sum().item() == pytest.approx(1)
    if C > 1:
        assert model.logits.grad.abs().max() > 0
    clone = copy.deepcopy(model)
    rng = torch.get_rng_state()
    a, _ = model.forward_objective()
    torch.set_rng_state(rng)
    b, _ = clone.forward_objective()
    assert torch.equal(a,b)


class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s_params = torch.nn.Parameter(torch.tensor([2.0]))

    def forward_objective(self):
        return (self.s_params - 0.5).square().sum(), {'w': torch.ones(1)}


@pytest.mark.parametrize('averaging', ['sum','running'])
def test_iterate_averaging_matches_notebook(averaging):
    expected = Quadratic()
    optimizer = torch.optim.Adam(expected.parameters(),lr=0.1)
    snapshots = []
    for i in range(8):
        optimizer.zero_grad()
        loss,_ = expected.forward_objective()
        loss.backward()
        optimizer.step()
        if i >= 5:
            snapshots.append(expected.s_params.detach().clone())
    if averaging == 'running':
        target = snapshots[0]
        for n,p in enumerate(snapshots[1:],2):
            target = (target*(n-1)+p)/n
    else:
        target = sum(snapshots, torch.zeros_like(snapshots[0]))/len(snapshots)
    actual,_ = fit_adam(Quadratic(),steps=8,lr=0.1,average_last=3,averaging=averaging)
    assert torch.equal(actual.s_params.detach(),target)
    assert not torch.equal(actual.s_params.detach(),snapshots[-1])


def test_current_toy_data_and_inducing_grid():
    data = prepare_toy_data(n_train=50,n_test=20)
    generator = torch.Generator().manual_seed(3)
    x = torch.rand(70,generator=generator)
    y = x+x**2+torch.randn(70,generator=generator)*(0.2*x+0.05)
    ids = torch.randperm(70,generator=torch.Generator().manual_seed(2))
    assert torch.equal(data['x'],x[ids[:50]])
    assert torch.equal(data['y_test'],y[ids[50:]])
    model = build_toy_model(data)
    assert model.C == 2
    assert model.m == 144
    assert model.beta == 1031.17
    assert float(model.Vu[:,-1].min()) == -0.05
    assert float(model.Vu[:,-1].max()) == 1.05
    first = model.predict_density(0.3, -0.3, 3.0, 100)
    assert all(np.isfinite(v).all() for v in first)


def test_goldberger_entropy_sign():
    torch.manual_seed(12)
    x = torch.linspace(0.1,0.9,30)
    m = LogisticGPMixtureDiag(x, x+0.1, base=StudentTLinearBase(1,0.4), C=3, m_induce=9)
    rng = torch.get_rng_state()
    _, corrected = m.forward_objective()
    m.entropy_correction = False
    torch.set_rng_state(rng)
    _, uncorrected = m.forward_objective()
    w = m._weights()
    torch.testing.assert_close(corrected['KL']-uncorrected['KL'], (w*torch.log(w+1e-12)).sum())


@pytest.mark.parametrize('kind', ['student', 'beta'])
def test_dtype_conversion_moves_cached_tensors(kind):
    x = torch.linspace(0.1, 0.9, 24).reshape(-1, 1)
    y = 0.2 + x[:, 0] / 2
    if kind == 'student':
        model = LogisticGPMixtureDiag(x, y, m_induce=9)
    else:
        model = BetaGPMixtureDiag(x, y, mu_base_train=np.full(24, 0.5), phi_base=5, m_induce=9)
    model.float()
    assert model.A.dtype == model.mus.dtype == torch.float32
    assert 'A' in dict(model.named_buffers())
    assert 'A' not in model.state_dict()
    assert torch.isfinite(model.forward_objective()[0])


def test_toy_model_pickle_preserves_prediction():
    model = build_toy_model(prepare_toy_data(n_train=30, n_test=10), m_induce=16)
    restored = pickle.loads(pickle.dumps(model))
    for a, b in zip(model.predict_density(0.3, M=100), restored.predict_density(0.3, M=100)):
        np.testing.assert_array_equal(a, b)


def test_public_kernel_params_in_builders(monkeypatch):
    from gpdr.examples import build_gini_model, build_weather_model
    import gpdr.examples as examples
    received = []

    class Capture:
        def __init__(self, x, y, **kwargs):
            received.append(kwargs['kp'])
        def to(self, device):
            return self

    monkeypatch.setattr(examples, 'GPDR', Capture)
    kp = KernelParams(log_lx2=math.log(0.2**2))
    build_gini_model({'x_t': torch.zeros(4, 5), 'y_t': torch.ones(4),
                      'mu_base_train': np.full(4, 0.5), 'phi_base': 5}, kp=kp)
    assert received[-1] is kp
    data = {'x_train': np.zeros((4, 3)), 'y_train': np.ones(4), 'x_test': np.zeros((2, 3)), 'y_test': np.ones(2),
            'to_z_from_y': None, 'base_g_terms': None, 'gam': None, 'ORIGINAL_SIGMA_NORMAL': 1.0}
    build_weather_model(data, kp=kp, device='cpu')
    assert received[-1] is kp
