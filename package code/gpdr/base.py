import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.stats import t as student_t


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

    def _params(self, y: torch.Tensor):
        mu = torch.as_tensor(self.mu, dtype=y.dtype, device=y.device)
        a = mu * self.phi
        b = (1.0 - mu) * self.phi
        return a, b

    def terms(self, y: torch.Tensor, x: torch.Tensor | None = None):
        a, b = self._params(y)
        return beta_derivatives(y, a, b)

    def cdf(self, y: torch.Tensor, x: torch.Tensor | None = None):
        a, b = self._params(y)
        return beta_cdf_torch(y, a, b)


def beta_pdf_torch(y, a, b):
    y = torch.clamp(y, 1e-12, 1.0 - 1e-12)
    return torch.exp((a - 1.0) * torch.log(y) + (b - 1.0) * torch.log1p(-y) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)))


def beta_cdf_torch(y, a, b):
    import scipy.special

    y_np = torch.clamp(y, 1e-12, 1.0 - 1e-12).detach().cpu().numpy()
    a_np = a.detach().cpu().numpy() if torch.is_tensor(a) else a
    b_np = b.detach().cpu().numpy() if torch.is_tensor(b) else b
    out = scipy.special.betainc(a_np, b_np, y_np)
    return torch.tensor(out, dtype=y.dtype, device=y.device)


def beta_derivatives(y, a, b):
    y = torch.clamp(y, 1e-8, 1.0 - 1e-8)
    g = beta_pdf_torch(y, a, b)
    score = (a - 1.0) / y - (b - 1.0) / (1.0 - y)
    score_prime = -(a - 1.0) / y**2 - (b - 1.0) / (1.0 - y) ** 2
    gy = g * score
    gyy = g * (score.square() + score_prime)
    return g, gy, gyy
