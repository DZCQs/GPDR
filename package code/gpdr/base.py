import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.stats import t as student_t


@dataclass
class FunctionalBase:
    """Use any fitted density via cdf(y, x), pdf(y, x), and terms(y, x).

    ``terms`` returns the density and its first two derivatives with respect to
    y. The functions must return tensors on y's device with y's shape. Fitting
    the base and transforming covariates remain the caller's responsibility.
    """

    cdf: Callable
    pdf: Callable
    terms: Callable

    def __reduce__(self):
        import cloudpickle
        return _restore_functional_base, (cloudpickle.dumps((self.cdf, self.pdf, self.terms)),)


def _restore_functional_base(payload):
    import cloudpickle
    return FunctionalBase(*cloudpickle.loads(payload))


def _as_tensor(value, like: torch.Tensor):
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


@dataclass
class StudentTLinearBase:
    slope: float
    scale: float
    df: float = 3.0

    def terms(self, y: torch.Tensor, x: torch.Tensor):
        b = _as_tensor(self.slope, x)
        sigma = _as_tensor(self.scale, x)
        mu = b * x.reshape_as(y)
        g_np = student_t.pdf(y.detach().cpu().numpy(), self.df, loc=mu.detach().cpu().numpy(), scale=float(sigma))
        g = torch.tensor(g_np, dtype=y.dtype, device=y.device)
        z = (y - mu) / sigma
        gy = -(self.df + 1.0) * z * g / (sigma * (self.df + z**2))
        gyy = g * (
            -(self.df + 1.0) / (sigma**2 * (self.df + z**2))
            + (self.df + 1.0) * z**2 * (self.df + 3.0) / (sigma**2 * (self.df + z**2) ** 2)
        )
        return g, gy, gyy

    def cdf(self, y: torch.Tensor, x: torch.Tensor):
        mu = self.slope * x.reshape_as(y)
        z_np = student_t.cdf(
            y.detach().cpu().numpy(),
            self.df,
            loc=mu.detach().cpu().numpy(),
            scale=self.scale,
        )
        return torch.tensor(z_np, dtype=y.dtype, device=y.device)

    def pdf(self, y: torch.Tensor, x: torch.Tensor):
        return self.terms(y, x)[0]


@dataclass
class StudentTLinearIndexBase:
    """Student-t base with location slope * (x @ coefficients)."""

    slope: float
    coefficients: np.ndarray | list[float]
    scale: float
    df: float = 3.0

    def _index(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        coeff = torch.as_tensor(self.coefficients, dtype=x.dtype, device=x.device)
        return x @ coeff.reshape(-1)

    def terms(self, y: torch.Tensor, x: torch.Tensor):
        b = _as_tensor(self.slope, y)
        sigma = _as_tensor(self.scale, y)
        mu = b * self._index(x)
        g_np = student_t.pdf(y.detach().cpu().numpy(), self.df, loc=mu.detach().cpu().numpy(), scale=float(sigma))
        g = torch.tensor(g_np, dtype=y.dtype, device=y.device)
        z = (y - mu) / sigma
        gy = -(self.df + 1.0) * z * g / (sigma * (self.df + z**2))
        gyy = g * (
            -(self.df + 1.0) / (sigma**2 * (self.df + z**2))
            + (self.df + 1.0) * z**2 * (self.df + 3.0) / (sigma**2 * (self.df + z**2) ** 2)
        )
        return g, gy, gyy

    def cdf(self, y: torch.Tensor, x: torch.Tensor):
        mu = self.slope * self._index(x)
        z_np = student_t.cdf(
            y.detach().cpu().numpy(),
            self.df,
            loc=mu.detach().cpu().numpy(),
            scale=self.scale,
        )
        return torch.tensor(z_np, dtype=y.dtype, device=y.device)

    def pdf(self, y: torch.Tensor, x: torch.Tensor):
        return self.terms(y, x)[0]


@dataclass
class StudentTGamBase:
    predict_fn: Callable[[np.ndarray], np.ndarray]
    scale: float
    df: float = 3.0

    def _mu(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        mu = self.predict_fn(x_np)
        return torch.tensor(mu, dtype=x.dtype, device=x.device).reshape(-1)

    def terms(self, y: torch.Tensor, x: torch.Tensor):
        sigma = _as_tensor(self.scale, y)
        mu = self._mu(x)
        g_np = student_t.pdf(y.detach().cpu().numpy(), self.df, loc=mu.detach().cpu().numpy(), scale=float(sigma))
        g = torch.tensor(g_np, dtype=y.dtype, device=y.device)
        z = (y - mu) / sigma
        gy = -(self.df + 1.0) * z * g / (sigma * (self.df + z**2))
        gyy = g * (
            -(self.df + 1.0) / (sigma**2 * (self.df + z**2))
            + (self.df + 1.0) * z**2 * (self.df + 3.0) / (sigma**2 * (self.df + z**2) ** 2)
        )
        return g, gy, gyy

    def cdf(self, y: torch.Tensor, x: torch.Tensor):
        mu = self._mu(x)
        z_np = student_t.cdf(y.detach().cpu().numpy(), self.df, loc=mu.detach().cpu().numpy(), scale=self.scale)
        return torch.tensor(z_np, dtype=y.dtype, device=y.device)

    def pdf(self, y: torch.Tensor, x: torch.Tensor):
        return self.terms(y, x)[0]


@dataclass
class BetaBase:
    mu: np.ndarray | float
    phi: float

    def __reduce__(self):
        import cloudpickle
        return _restore_beta_base, (cloudpickle.dumps(self.mu), self.phi)

    def _params(self, y: torch.Tensor, x=None):
        mu = self.mu(x) if callable(self.mu) else self.mu
        mu = torch.as_tensor(mu, dtype=y.dtype, device=y.device)
        a = mu * self.phi
        b = (1.0 - mu) * self.phi
        return a, b

    def terms(self, y: torch.Tensor, x: torch.Tensor | None = None):
        if torch.any((y <= 0) | (y >= 1)):
            raise ValueError("beta score derivatives require responses strictly inside (0, 1)")
        a, b = self._params(y, x)
        return beta_derivatives(y, a, b)

    def cdf(self, y: torch.Tensor, x: torch.Tensor | None = None):
        a, b = self._params(y, x)
        return beta_cdf_torch(y, a, b)

    def pdf(self, y: torch.Tensor, x: torch.Tensor | None = None):
        a, b = self._params(y, x)
        return beta_pdf_torch(y, a, b)


def beta_pdf_torch(y, a, b):
    original = y
    y = torch.where((y <= 0) | (y >= 1), torch.full_like(y, 0.5), y)
    density = torch.exp((a - 1.0) * torch.log(y) + (b - 1.0) * torch.log1p(-y) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)))
    left = torch.where(a < 1, torch.full_like(density, float('inf')), torch.where(a == 1, b, 0.))
    right = torch.where(b < 1, torch.full_like(density, float('inf')), torch.where(b == 1, a, 0.))
    density = torch.where(original == 0, left, density)
    density = torch.where(original == 1, right, density)
    return torch.where((original < 0) | (original > 1), torch.zeros_like(density), density)


def _restore_beta_base(mu, phi):
    import cloudpickle
    return BetaBase(cloudpickle.loads(mu), phi)


def beta_cdf_torch(y, a, b):
    import scipy.special

    y_np = torch.clamp(y, 0.0, 1.0).detach().cpu().numpy()
    a_np = a.detach().cpu().numpy() if torch.is_tensor(a) else a
    b_np = b.detach().cpu().numpy() if torch.is_tensor(b) else b
    out = scipy.special.betainc(a_np, b_np, y_np)
    return torch.tensor(out, dtype=y.dtype, device=y.device)


def beta_derivatives(y, a, b):
    g = beta_pdf_torch(y, a, b)
    score = (a - 1.0) / y - (b - 1.0) / (1.0 - y)
    score_prime = -(a - 1.0) / y**2 - (b - 1.0) / (1.0 - y) ** 2
    gy = g * score
    gyy = g * (score.square() + score_prime)
    return g, gy, gyy
