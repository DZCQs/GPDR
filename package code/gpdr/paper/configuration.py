"""Data/base-model configuration for the paper, not GPDR implementations.

Each function returns arguments for the public GPDR constructor. The same
constructor, objective, optimizer, and predictor are used for new datasets.
"""
import math
import numpy as np
import torch

from ..base import BetaBase, FunctionalBase
from ..inducing import kmeans_inducing, sample_inducing, tensor_grid
from ..kernels import KernelParams


TOY_FIT = dict(steps=300000, lr=0.5, average_last=5000,
               averaging="running", trace_at="before", verbose_every=10000)
WEATHER_FIT = dict(steps=5000, lr=0.5, average_last=500,
                   averaging="sum", trace_at="after", verbose_every=500)
GINI_FIT = dict(steps=2000, lr=0.5, trace_at="after", verbose_every=200)


def toy_model_inputs(data, *, C=2, m_induce=150, beta=1031.17, kp=None):
    x, y = data['x'], data['y']
    n_axis = int(math.sqrt(m_induce))
    t_pdf, slope, scale, df = data['t_pdf'], data['B_T'], data['SCALE_T'], data['DF']
    base = FunctionalBase(
        cdf=data['to_z_from_y'], terms=data['base_g_terms'],
        pdf=lambda y, x: t_pdf(y, slope * x, scale, df))
    return dict(x=x, y=y, base=base, C=C, beta=beta,
                inducing_points=tensor_grid(
                    torch.linspace(x.min(), x.max(), n_axis, dtype=x.dtype, device=x.device),
                    torch.linspace(-0.05, 1.05, n_axis, dtype=x.dtype, device=x.device)),
                kp=kp or KernelParams(math.log(0.9199**2), math.log(0.0786**2), math.log(0.3363**2)))


def weather_model_inputs(data, *, C=2, m_induce=1000, beta=29.6, kp=None):
    x, y = data['x_train'], data['y_train']
    per_axis = int(round(m_induce ** (1 / (x.shape[1] + 1))))
    base_terms = data['base_g_terms']
    gam, sigma = data['gam'], data['ORIGINAL_SIGMA_NORMAL']
    base = FunctionalBase(cdf=data['to_z_from_y'], terms=base_terms,
                           pdf=lambda y, x: base_terms(y, x)[0])
    def original_pdf(y, x):
        mu = torch.tensor(gam.predict(x.cpu().numpy()), device=x.device, dtype=x.dtype)
        return 1.0 / (math.sqrt(2.0 * math.pi) * sigma) * torch.exp(-0.5 * ((y - mu) / sigma) ** 2)
    reference = FunctionalBase(cdf=None, terms=None, pdf=original_pdf)
    # The notebook omits the first covariate center. Keep that explicit here;
    # the general-purpose k-means helper returns every center.
    return dict(x=x, y=y, base=base, reference_base=reference, C=C, beta=beta,
                inducing_points=lambda V: kmeans_inducing(V, m_induce)[per_axis:],
                normalization_size=x.numel(), loss_reduction="sum",
                warm_start_solver="solve", exp_clip=None,
                kp=kp or KernelParams(math.log(0.1806**2),
                    torch.tensor([math.log(0.4901**2)] * x.shape[1], dtype=x.dtype, device=x.device),
                    math.log(0.236**2)))


def gini_model_inputs(data, *, C=2, m_induce=200, beta=493.61, seed=2, kp=None):
    x, y = data['x_t'], data['y_t']
    return dict(x=x, y=y, base=BetaBase(data['mu_base_train'], data['phi_base']),
                inducing_points=lambda V: sample_inducing(V, m_induce, seed),
                C=C, beta=beta, train_pit_epsilon=1e-6,
                kp=kp or KernelParams(math.log(0.5769**2),
                    torch.log(torch.full((x.shape[1],), 0.507**2, dtype=x.dtype, device=x.device)),
                    math.log(0.8355**2)))


def predict_gini_row(model, data, *, x_star_unscaled=None, X_row=None, df_row=None,
                     y_min=0.001, y_max=0.999, M=600):
    """Build kernel/base covariates for a dataframe row, then use GPDR.predict_density."""
    if X_row is None:
        raise ValueError("Pass the base-regression design row X_row")
    xvec = torch.from_numpy(X_row.to_numpy(dtype=float)).to(model.V.dtype).to(model.V.device)
    eta = torch.from_numpy(data['beta_hat']).to(model.V.dtype).to(model.V.device) @ xvec
    mu = torch.sigmoid(eta)
    row = df_row if df_row is not None else x_star_unscaled
    xs = np.array([row[c] for c in data['gp_cols']], dtype=float)
    xs = (xs - data['x_min'].to_numpy()) / (data['x_max'].to_numpy() - data['x_min'].to_numpy() + 1e-12)
    return model.predict_density(xs, y_min=y_min, y_max=y_max, M=M,
        base=BetaBase(mu, data['phi_base']),
        reference_base=BetaBase(mu, data['phi_base_original']))
