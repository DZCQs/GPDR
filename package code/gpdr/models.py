import math

import numpy as np
import torch
import torch.nn as nn

from .base import BetaBase, StudentTLinearBase, beta_cdf_torch, beta_derivatives, beta_pdf_torch
from .kernels import KernelParams, RBFSEARD, chol_inv


def _grid_inducing(V, m_induce):
    d = V.shape[1]
    if d == 2:
        s = int(math.sqrt(m_induce))
        gx = torch.linspace(V[:, 0].min(), V[:, 0].max(), s, dtype=V.dtype, device=V.device)
        gz = torch.linspace(1e-3, 1.0 - 1e-3, s, dtype=V.dtype, device=V.device)
        mesh = torch.meshgrid(gx, gz, indexing="ij")
        return torch.stack([mesh[0].reshape(-1), mesh[1].reshape(-1)], dim=1)
    n = min(m_induce, V.shape[0])
    return V[torch.linspace(0, V.shape[0] - 1, n, dtype=torch.long, device=V.device)].clone()


def _unit_grid_inducing(d_x, m_induce, *, dtype, device, epsilon=1e-6):
    """Tensor-product grid on [epsilon, 1-epsilon]^d_x x [epsilon, 1-epsilon]."""

    n_per_dim = max(3, int(round(m_induce ** (1.0 / (d_x + 1)))))
    axes = [torch.linspace(epsilon, 1.0 - epsilon, n_per_dim, dtype=dtype, device=device) for _ in range(d_x + 1)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in mesh], dim=1)


def _sample_inducing(V, m_induce, seed=2):
    ggen = torch.Generator(device=V.device).manual_seed(seed)
    idx = torch.randperm(V.shape[0], generator=ggen, device=V.device)[:m_induce]
    return V[idx].clone()


def _kmeans_inducing(x, z, m_induce):
    try:
        from kmeans_pytorch import kmeans
    except Exception as exc:
        raise ImportError("kmeans_pytorch is required for weather-style inducing points") from exc

    dim_x = x.shape[1]
    n_per_dim = int(round(m_induce ** (1.0 / (dim_x + 1))))
    total_x = n_per_dim**dim_x
    _, centers = kmeans(X=x, num_clusters=total_x, distance="euclidean", device=x.device)
    centers = torch.as_tensor(centers, dtype=x.dtype, device=x.device)
    z_grid = torch.linspace(1e-5, 1.0 - 1e-5, n_per_dim, dtype=x.dtype, device=x.device)
    x_grid = centers[:, None, :].expand(centers.size(0), n_per_dim, dim_x)
    z_expand = z_grid[None, :, None].expand(centers.size(0), n_per_dim, 1)
    return torch.cat([x_grid, z_expand], dim=2).reshape(-1, dim_x + 1)


class _GPDRBase(nn.Module):
    def _objective_parts(self, weights):
        aMu = (self.A @ self.mus.T).T
        mu_bar = torch.sum(weights[:, None] * self.mus, dim=0)
        Ez = self.A @ mu_bar
        Efzz = self.B @ mu_bar
        g, gy, gyy = self.g, self.gy, self.gyy
        c = gy / g
        const = (gyy / g) - 0.5 * c.square()
        lin_coeff = 2.0 * gy
        v_list = torch.exp(2.0 * self.s_params)
        row_var = (self.A2 @ v_list.T).T
        Efz2 = torch.sum(weights[:, None] * (row_var + aMu.square()), dim=0)
        EH = torch.mean(const + lin_coeff * Ez + 0.5 * g.square() * Efz2 + g.square() * Efzz)

        Kinv = self.K11uu_inv
        logdetK = 2.0 * torch.log(torch.diag(self.K11uu_chol)).sum() if self.K11uu_chol is not None else torch.slogdet(self.K11uu)[1]
        diag_Kinv = torch.diag(Kinv)
        KL_k = []
        for k in range(self.C):
            mu_k = self.mus[k]
            v_k = v_list[k]
            quad = mu_k @ (Kinv @ mu_k)
            tr = torch.dot(diag_Kinv, v_k)
            logdetSigma = 2.0 * torch.sum(self.s_params[k])
            KL_k.append(0.5 * (tr + quad - self.m + logdetK - logdetSigma))
        KL = torch.sum(weights * torch.stack(KL_k))
        return KL, EH, Ez, Efzz

    @torch.no_grad()
    def _f_on_query(self, Vq):
        K01 = self.kern.delta_01(Vq, self.Vu)
        weights = self._weights()
        m_tilde = torch.sum(weights[:, None] * self.mus, dim=0)
        return K01 @ (self.K11uu_inv @ m_tilde)


class LogisticGPMixtureDiag(_GPDRBase):
    def __init__(
        self,
        x,
        y,
        *,
        base=None,
        C=2,
        m_induce=121,
        beta=1.0,
        seed=2,
        kp: KernelParams = KernelParams(),
        inducing="grid",
        scale_hyvarinen_by_n=False,
    ):
        super().__init__()
        self.x = x.detach().clone()
        self.y = y.detach().clone()
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        self.n = self.x.numel()
        self.C = C
        self.beta = beta
        self.scale_hyvarinen_by_n = scale_hyvarinen_by_n
        self.base_model = base if base is not None else StudentTLinearBase(1.0, 1.0, 3.0)
        self.kern = RBFSEARD(kp.with_dim(self.x.shape[1], dtype=self.x.dtype, device=self.x.device), self.x.shape[1], dtype=self.x.dtype, device=self.x.device)

        z = torch.clamp(self.base_model.cdf(self.y, self.x), 1e-5, 1.0 - 1e-5)
        self.V = torch.cat([self.x, z.reshape(-1, 1)], dim=1)
        if inducing == "kmeans":
            self.Vu = _kmeans_inducing(self.x, z, m_induce)
        elif inducing == "unit_grid":
            self.Vu = _unit_grid_inducing(self.x.shape[1], m_induce, dtype=self.x.dtype, device=self.x.device)
        else:
            self.Vu = _grid_inducing(self.V, m_induce)
        self.m = self.Vu.shape[0]
        self.g, self.gy, self.gyy = self.base_model.terms(self.y, self.x)
        self.K11uu = self.kern.delta_11(self.Vu, self.Vu)
        self.K11uu_inv, self.K11uu_chol = chol_inv(self.K11uu)
        self.A = self.kern.delta_11(self.V, self.Vu) @ self.K11uu_inv
        self.B = self.kern.delta_21(self.V, self.Vu) @ self.K11uu_inv
        self.A2 = self.A.square()
        self.logits = nn.Parameter(torch.randn(C, dtype=self.x.dtype, device=self.x.device))
        self.mus = nn.Parameter(torch.randn(C, self.m, dtype=self.x.dtype, device=self.x.device))
        self.s_params = nn.Parameter(torch.full((C, self.m), math.log(0.35), dtype=self.x.dtype, device=self.x.device))
        self.warm_start_mu(seed)

    def _weights(self):
        return torch.softmax(self.logits, dim=0)

    def warm_start_mu(self, seed=2):
        lin = 2.0 * self.gy + 2.0 * (self.gy / self.g) * self.g
        AtW = self.A.T * (self.g.square() / self.n)
        H = self.K11uu_inv + 2.0 * self.beta * (AtW @ self.A)
        b = self.beta * (self.A.T @ (lin / self.n))
        try:
            L = torch.linalg.cholesky(H)
            mu_ls = -torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
        except Exception:
            mu_ls = -torch.linalg.solve(H, b)
        with torch.no_grad():
            for k in range(self.C):
                self.mus[k].copy_(mu_ls / self.C + 0.03 * torch.randn_like(mu_ls))

    def forward_objective(self):
        KL, EH, _, _ = self._objective_parts(self._weights())
        scale_by_n = getattr(self, "scale_hyvarinen_by_n", self.x.ndim == 2 and self.x.shape[1] > 1)
        weighted = self.beta * (self.n if scale_by_n else 1.0) * EH
        label = "normalized EH" if scale_by_n else "EH"
        return KL + weighted, {"KL": KL.detach(), label: weighted.detach(), "w": self._weights().detach()}

    @torch.no_grad()
    def predict_density(self, x_star, y_min, y_max, M=700):
        xs = torch.as_tensor(x_star, dtype=self.x.dtype, device=self.x.device).reshape(1, -1)
        y_grid = torch.linspace(float(y_min), float(y_max), M, dtype=self.x.dtype, device=self.x.device)
        Xq = xs.repeat(M, 1)
        z_grid = torch.clamp(self.base_model.cdf(y_grid, Xq), 1e-6, 1.0 - 1e-6)
        Vq = torch.cat([Xq, z_grid.reshape(-1, 1)], dim=1)
        f_vals = self._f_on_query(Vq)
        g_vals = self.base_model.pdf(y_grid, Xq)
        idx = torch.argsort(z_grid)
        z_sorted, f_sorted = z_grid[idx], f_vals[idx]
        dz = z_sorted[1:] - z_sorted[:-1]
        wts = torch.zeros_like(z_sorted)
        wts[0], wts[-1] = dz[0] / 2.0, dz[-1] / 2.0
        wts[1:-1] = (dz[:-1] + dz[1:]) / 2.0
        exp_f = torch.exp(torch.clamp(f_sorted, -50, 50))
        denom = torch.sum(exp_f * wts) + 1e-12
        h_hat = g_vals * torch.exp(torch.clamp(f_vals, -50, 50)) / denom
        return y_grid.cpu().numpy(), g_vals.cpu().numpy(), h_hat.cpu().numpy(), exp_f.cpu().numpy(), z_grid.cpu().numpy()


class BetaGPMixtureDiag(_GPDRBase):
    def __init__(self, x, y, *, mu_base_train, phi_base, C=1, m_induce=250, beta=50.0, seed=2, kp: KernelParams = KernelParams()):
        super().__init__()
        self.x = x.detach().clone()
        self.y = y.detach().clone()
        self.n, self.d_x = self.x.shape
        self.C = C
        self.beta = beta
        self.mu_base_train = np.asarray(mu_base_train)
        self.phi_base = float(phi_base)
        self.kern = RBFSEARD(kp.with_dim(self.d_x, dtype=self.x.dtype, device=self.x.device), self.d_x, dtype=self.x.dtype, device=self.x.device)

        mu_t = torch.from_numpy(self.mu_base_train).to(dtype=self.x.dtype, device=self.x.device)
        a_t = mu_t * self.phi_base
        b_t = (1.0 - mu_t) * self.phi_base
        self.g, self.gy, self.gyy = beta_derivatives(self.y, a_t, b_t)
        z_t = torch.clamp(beta_cdf_torch(self.y, a_t, b_t), 1e-6, 1.0 - 1e-6)
        self.V = torch.cat([self.x, z_t.reshape(-1, 1)], dim=1)
        self.Vu = _sample_inducing(self.V, m_induce, seed)
        self.m = self.Vu.shape[0]
        self.K11uu = self.kern.delta_11(self.Vu, self.Vu)
        self.K11uu_inv, self.K11uu_chol = chol_inv(self.K11uu)
        self.A = self.kern.delta_11(self.V, self.Vu) @ self.K11uu_inv
        self.B = self.kern.delta_21(self.V, self.Vu) @ self.K11uu_inv
        self.A2 = self.A.square()
        self.register_buffer("w_fixed", torch.full((C,), 1.0 / float(C), dtype=self.x.dtype, device=self.x.device))
        self.mus = nn.Parameter(torch.randn(C, self.m, dtype=self.x.dtype, device=self.x.device))
        self.s_params = nn.Parameter(torch.full((C, self.m), math.log(0.35), dtype=self.x.dtype, device=self.x.device))
        self.warm_start_mu(seed)

    def _weights(self):
        return self.w_fixed

    def warm_start_mu(self, seed=2):
        lin = 2.0 * self.gy + 2.0 * (self.gy / self.g) * self.g
        AtW = self.A.T * (self.g.square() / self.n)
        H = self.K11uu_inv + 2.0 * self.beta * (AtW @ self.A)
        b = self.beta * (self.A.T @ (lin / self.n))
        try:
            L = torch.linalg.cholesky(H)
            mu_ls = -torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
        except Exception:
            mu_ls = -torch.linalg.solve(H, b)
        with torch.no_grad():
            for k in range(self.C):
                self.mus[k].copy_(mu_ls / max(self.C, 1) + 0.03 * torch.randn_like(mu_ls))

    def forward_objective(self):
        KL, EH, _, _ = self._objective_parts(self.w_fixed)
        return KL + self.beta * EH, {"KL": KL.detach(), "EH": EH.detach(), "w": self.w_fixed.detach()}

    @torch.no_grad()
    def predict_density_scaled(self, x_scaled, mu_star, phi_star, y_min=1e-3, y_max=1 - 1e-3, M=600):
        xs = torch.as_tensor(x_scaled, dtype=self.x.dtype, device=self.x.device).reshape(1, -1)
        y_grid = torch.linspace(y_min, y_max, M, dtype=self.x.dtype, device=self.x.device)
        a_star = torch.as_tensor(mu_star * phi_star, dtype=self.x.dtype, device=self.x.device)
        b_star = torch.as_tensor((1.0 - mu_star) * phi_star, dtype=self.x.dtype, device=self.x.device)
        g_vals = beta_pdf_torch(y_grid, a_star, b_star)
        z_grid = torch.clamp(beta_cdf_torch(y_grid, a_star, b_star), 1e-6, 1.0 - 1e-6)
        Vq = torch.cat([xs.repeat(M, 1), z_grid.reshape(-1, 1)], dim=1)
        f_vals = self._f_on_query(Vq)
        idx = torch.argsort(z_grid)
        z_sorted, f_sorted = z_grid[idx], f_vals[idx]
        dz = z_sorted[1:] - z_sorted[:-1]
        wts = torch.zeros_like(z_sorted)
        wts[0], wts[-1] = dz[0] / 2.0, dz[-1] / 2.0
        wts[1:-1] = (dz[:-1] + dz[1:]) / 2.0
        exp_f = torch.exp(torch.clamp(f_sorted, -50, 50))
        denom = torch.sum(exp_f * wts) + 1e-12
        h_hat = g_vals * torch.exp(torch.clamp(f_vals, -50, 50)) / denom
        return y_grid.cpu().numpy(), g_vals.cpu().numpy(), h_hat.cpu().numpy(), torch.exp(torch.clamp(f_vals, -50, 50)).cpu().numpy(), z_grid.cpu().numpy()


# Notebook pickle compatibility names.
RBF_SE_ARD = RBFSEARD
