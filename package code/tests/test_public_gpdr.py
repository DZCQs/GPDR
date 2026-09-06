"""The supported public interface, including data unlike the paper examples."""
import ast
import math
import pickle
from pathlib import Path
import numpy as np
import pytest
import torch

from gpdr import GPDR, FunctionalBase, BetaBase, KernelParams, fit_adam, sample_inducing, evaluate_predictions
from gpdr import density_mean_interval
from gpdr.paper import toy, weather, gini


def test_one_implementation_for_all_examples():
    for module in (toy, weather, gini):
        assert module.GPDR is GPDR
        assert module.fit_adam is fit_adam
        tree = ast.parse(Path(module.__file__).read_text())
        assert not any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        assert not any(isinstance(n, ast.Attribute) and n.attr == 'backward' for n in ast.walk(tree))
    tree = ast.parse(Path(__import__('gpdr.models', fromlist=['']).__file__).read_text())
    assert sum(isinstance(n, ast.FunctionDef) and n.name == 'forward_objective' for n in ast.walk(tree)) == 1
    for path in Path(toy.__file__).parent.glob('*_model.py'):
        pytest.fail(f'Duplicate paper model implementation: {path}')


def test_new_nonlinear_three_covariate_base():
    torch.manual_seed(37)
    x = torch.rand(60, 3, dtype=torch.float64)
    def mu(x):
        return torch.sin(x[:, 0]) + x[:, 1] * x[:, 2]
    def pdf(y, x):
        return torch.exp(-0.5 * ((y-mu(x))/0.4)**2) / (math.sqrt(2*math.pi)*0.4)
    def cdf(y, x):
        return torch.distributions.Normal(mu(x), 0.4).cdf(y)
    def terms(y, x):
        g = pdf(y, x)
        s = -(y-mu(x))/0.4**2
        return g, g*s, g*(s**2-1/0.4**2)
    base = FunctionalBase(cdf, pdf, terms)
    y = mu(x) + 0.4*torch.randn(60, dtype=x.dtype)
    model = GPDR(x, y, base=base, inducing_points=lambda V: sample_inducing(V, 12),
                 C=3, beta=2., kp=KernelParams(math.log(0.4**2), math.log(0.3**2), math.log(0.5**2)))
    model, trace = fit_adam(model, steps=5, lr=0.01)
    assert type(model) is GPDR and len(trace['F']) == 5
    predictions = [model.predict_density(row, -3., 4., M=500) for row in x[:4]]
    assert all(abs(np.trapezoid(p.density, p.y)-1) < 0.002 for p in predictions)
    metrics = evaluate_predictions(y[:4].numpy(), predictions)
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics['mean_log_score'] == metrics['log_score']/4
    torch.testing.assert_close(model._weights().sum(), torch.ones((), dtype=x.dtype))


def test_conditional_beta_and_reference_do_not_change_h():
    x = torch.linspace(0.1, 0.9, 30, dtype=torch.float64).reshape(-1,1)
    y = 0.2 + 0.4*x[:,0]
    mean = lambda x: torch.sigmoid(x[:,0]-0.5)
    model = GPDR(x,y,base=BetaBase(mean,5.),C=2,
                 inducing_points=lambda V: sample_inducing(V,8),train_pit_epsilon=1e-6)
    a = model.predict_density([0.3],0.001,0.999,M=101)
    b = model.predict_density([0.3],0.001,0.999,M=101,reference_base=BetaBase(mean,12.))
    np.testing.assert_array_equal(a.density,b.density)
    assert not np.array_equal(a.reference_density,b.reference_density)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(a.density, restored.predict_density([0.3],0.001,0.999,M=101).density)


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_beta_support(dtype):
    base = BetaBase(0.5,2.)
    y = torch.tensor([-0.2,0.,0.3,1.,1.2],dtype=dtype)
    torch.testing.assert_close(base.pdf(y),torch.tensor([0.,1.,1.,1.,0.],dtype=dtype))
    torch.testing.assert_close(base.cdf(y),torch.tensor([0.,0.,0.3,1.,1.],dtype=dtype))


def test_beta_interior_is_not_clipped():
    base = BetaBase(0.5,4.)
    y = torch.tensor([1e-8,1e-10,0.3],dtype=torch.float32)
    expected = 6*y*(1-y)
    torch.testing.assert_close(base.pdf(y),expected,rtol=2e-6,atol=0.)
    torch.testing.assert_close(base.terms(y)[0],expected,rtol=2e-6,atol=0.)


def test_checkpoint_requires_same_inducing_basis_and_kernel():
    x = torch.linspace(0.1,0.9,20,dtype=torch.float64).reshape(-1,1)
    y = 0.2+0.4*x[:,0]
    kwargs = dict(base=BetaBase(0.5,4.))
    a = GPDR(x,y,inducing_points=lambda V:sample_inducing(V,5,seed=2),**kwargs)
    checkpoint = a.state_dict()
    b = GPDR(x,y,inducing_points=lambda V:sample_inducing(V,5,seed=3),**kwargs)
    with pytest.raises(ValueError,match='Vu differs'):
        b.load_state_dict(checkpoint)
    with pytest.raises(ValueError,match='Vu differs'):
        torch.nn.ModuleDict({'model':b}).load_state_dict(torch.nn.ModuleDict({'model':a}).state_dict())
    b = GPDR(x,y,inducing_points=checkpoint['Vu'],**kwargs)
    b.load_state_dict(checkpoint)
    np.testing.assert_array_equal(a.predict_density([0.3],0.001,0.999,M=51).density,
                                  b.predict_density([0.3],0.001,0.999,M=51).density)
    bad = dict(checkpoint, **{'kern.lz2':checkpoint['kern.lz2']*2})
    with pytest.raises(ValueError,match='kern.lz2 differs'):
        b.load_state_dict(bad)


def test_uniform_grid_quantiles():
    mean,lo,hi = density_mean_interval(np.array([0.,0.5,1.]), np.ones(3), level=0.5)
    assert (mean,lo,hi) == (0.5,0.25,0.75)


@pytest.mark.parametrize('kwargs', [dict(normalization_size=0),dict(normalization_size=-1),
                                  dict(beta=float('nan')),dict(prediction_pit_epsilon=0.5),
                                  dict(exp_clip=-1.)])
def test_invalid_numerics_rejected(kwargs):
    x = torch.zeros(10,2,dtype=torch.float64)
    y = torch.full((10,),0.4,dtype=torch.float64)
    with pytest.raises(ValueError):
        GPDR(x,y,base=BetaBase(0.5,4.),inducing_points=lambda V:sample_inducing(V,4),**kwargs)


@pytest.mark.parametrize('n', [2,3,5])
def test_new_data_dimensions(n):
    x = torch.zeros(20,n,dtype=torch.float64)
    y = torch.linspace(0.2,0.8,20,dtype=torch.float64)
    model = GPDR(x,y,base=BetaBase(0.5,4.), inducing_points=lambda V: sample_inducing(V,5))
    p = model.predict_density(np.zeros(n),0.001,0.999,M=51)
    assert p.density.shape == (51,) and np.isfinite(p.density).all()
