import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class KernelParams:
    log_sigma2: float = math.log(0.3323**2)
    log_lx2: float | torch.Tensor = math.log(0.1633**2)
    log_lz2: float = math.log(0.2312**2)

    @property
    def sigma2(self) -> float:
        return math.exp(float(self.log_sigma2))

    @property
    def lx2(self):
        if torch.is_tensor(self.log_lx2):
            return torch.exp(self.log_lx2)
        return math.exp(float(self.log_lx2))

    @property
    def lz2(self) -> float:
        return math.exp(float(self.log_lz2))

    def with_dim(self, d_x: int, *, dtype=None, device=None) -> "KernelParams":
        if torch.is_tensor(self.log_lx2):
            lx = self.log_lx2.to(dtype=dtype, device=device)
        else:
            lx = torch.full((d_x,), float(self.log_lx2), dtype=dtype, device=device)
        return KernelParams(self.log_sigma2, lx, self.log_lz2)


class RBFSEARD(nn.Module):
    """Squared-exponential ARD kernel with z-derivative covariance blocks."""

    def __init__(self, kp: KernelParams, d_x: int | None = None, *, device=None, dtype=None):
        super().__init__()
        if d_x is None:
            d_x = 1
        lx2 = kp.lx2
        if not torch.is_tensor(lx2):
            lx2 = torch.full((d_x,), float(lx2), dtype=dtype, device=device)
        else:
            lx2 = lx2.to(dtype=dtype, device=device)
        self.register_buffer("lx2", lx2)
        self.register_buffer("lz2", torch.tensor(kp.lz2, dtype=dtype, device=device))
        self.register_buffer("sigma2", torch.tensor(kp.sigma2, dtype=dtype, device=device))

    def _split(self, V, U):
        return V[:, :-1], V[:, -1:], U[:, :-1], U[:, -1:].T

    def _base(self, V, U):
        X1, z1, X2, z2 = self._split(V, U)
        dx2 = ((X1[:, None, :] - X2[None, :, :]) ** 2 / self.lx2).sum(-1)
        dz = z1 - z2
        K = self.sigma2 * torch.exp(-(dx2 + dz.square() / self.lz2))
        return K, dz

    def delta_01(self, V, U):
        K, dz = self._base(V, U)
        return K * (2.0 * dz / self.lz2)

    def delta_11(self, V, U):
        K, dz = self._base(V, U)
        dz2 = dz.square() / self.lz2
        out = (2.0 / self.lz2) * K * (1.0 - 2.0 * dz2)
        if V.shape == U.shape and torch.allclose(V, U):
            out = out + 1e-10 * torch.eye(V.shape[0], dtype=V.dtype, device=V.device)
        return out

    def delta_21(self, V, U):
        K, dz = self._base(V, U)
        return K * (8.0 * dz**3 / self.lz2**3 - 12.0 * dz / self.lz2**2)


# Notebook pickle compatibility names.
RBF_SE_ARD = RBFSEARD


def chol_inv(K: torch.Tensor):
    try:
        L = torch.linalg.cholesky(K)
        I = torch.eye(K.size(-1), dtype=K.dtype, device=K.device)
        if K.device.type == "mps":
            Y = torch.linalg.solve_triangular(L, I, upper=False)
            Kinv = torch.linalg.solve_triangular(L.T, Y, upper=True)
        else:
            Kinv = torch.cholesky_solve(I, L)
        return Kinv, L
    except Exception:
        U, S, Vt = torch.linalg.svd(K)
        threshold = 1e-10 * S[0]
        S_inv = torch.where(S > threshold, 1.0 / S, 0.0)
        return (Vt.T * S_inv.unsqueeze(0)) @ Vt, None

