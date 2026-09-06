"""One variational GPDR implementation for arbitrary conditional base models."""
import math
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from .base import BetaBase, StudentTLinearBase
from .inducing import kmeans_inducing, sample_inducing, tensor_grid
from .kernels import KernelParams, RBFSEARD, chol_inv


class DensityPrediction(NamedTuple):
    y: np.ndarray
    base_density: np.ndarray
    reference_density: np.ndarray
    density: np.ndarray
    correction: np.ndarray
    pit: np.ndarray


class GPDR(nn.Module):
    """GP log-density correction with a caller-supplied base distribution.

    x has shape (n, d), or (n,) for one covariate; y has shape (n,).
    base implements cdf(y, x), pdf(y, x), and terms(y, x) -> (g, gy, gyy).
    inducing_points is a tensor of (S(x), z) rows or a callable V -> Vu.
    Pass S(x) as x: no dataset-dependent transformation is hidden in the model.

    Numerical options explicitly configure score scaling, warm-start solver,
    PIT clipping, and exponentiation. None dispatch on an example name.
    """

    def __init__(self, x, y, *, base, inducing_points, kp=None, C=2, beta=1.0,
                 gumbel_tau=0.5, imputation="gumbel", entropy_correction=True,
                 normalization_size=None, loss_reduction="mean",
                 warm_start_solver="cholesky", train_pit_epsilon=1e-5,
                 prediction_pit_epsilon=1e-6, exp_clip=50.0,
                 reference_base=None):
        super().__init__()
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise TypeError("x and y must be floating-point torch tensors")
        if x.ndim not in (1, 2) or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("require x.shape=(n,d) or (n,), y.shape=(n,)")
        if x.dtype != y.dtype or x.device != y.device or not x.is_floating_point():
            raise ValueError("x and y must share a floating-point dtype and device")
        if not isinstance(C, int) or C < 1 or not math.isfinite(beta) or beta <= 0 or not math.isfinite(gumbel_tau) or gumbel_tau <= 0:
            raise ValueError("C, beta, and gumbel_tau must be positive")
        if normalization_size is not None and (not math.isfinite(normalization_size) or normalization_size <= 0):
            raise ValueError("normalization_size must be finite and positive")
        if any(not math.isfinite(e) or not 0 < e < 0.5 for e in (train_pit_epsilon, prediction_pit_epsilon)):
            raise ValueError("PIT clipping epsilons must lie strictly between 0 and 0.5")
        if exp_clip is not None and (not math.isfinite(exp_clip) or exp_clip <= 0):
            raise ValueError("exp_clip must be None or finite and positive")
        if imputation not in {"gumbel", "mean"}:
            raise ValueError("imputation must be 'gumbel' or 'mean'")
        if loss_reduction not in {"mean", "sum"}:
            raise ValueError("loss_reduction must be 'mean' or 'sum'")
        if warm_start_solver not in {"cholesky", "solve"}:
            raise ValueError("warm_start_solver must be 'cholesky' or 'solve'")
        self.register_buffer("x", x.detach().clone(), persistent=False)
        self.register_buffer("y", y.detach().clone(), persistent=False)
        self.d_x = x.shape[1] if x.ndim == 2 else 1
        self.n = x.shape[0] if normalization_size is None else normalization_size
        self.C, self.beta = C, beta
        self.base_model, self.reference_base = base, reference_base
        self.gumbel_tau, self.imputation = gumbel_tau, imputation
        self.entropy_correction = entropy_correction
        self.loss_reduction = loss_reduction
        self.warm_start_solver = warm_start_solver
        self.prediction_pit_epsilon = prediction_pit_epsilon
        self.exp_clip = exp_clip
        self.kern = RBFSEARD(kp or KernelParams(), self.d_x, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            z = base.cdf(self.y, self.x).clamp(train_pit_epsilon, 1 - train_pit_epsilon)
            V = torch.cat([self.x.reshape(-1, self.d_x), z.reshape(-1, 1)], dim=1)
            self.register_buffer("V", V, persistent=False)
            Vu = inducing_points(V) if callable(inducing_points) else inducing_points
            Vu = torch.as_tensor(Vu, dtype=x.dtype, device=x.device)
            if Vu.ndim != 2 or Vu.shape[1] != self.d_x + 1 or not len(Vu):
                raise ValueError("inducing_points must have shape (m, d+1)")
            self.register_buffer("Vu", Vu.detach().clone())
            self.m = len(Vu)
            for name, value in zip(("g", "gy", "gyy"), base.terms(self.y, self.x)):
                if value.shape != self.y.shape:
                    raise ValueError("base.terms must return tensors with y.shape")
                self.register_buffer(name, value, persistent=False)
            K11uu = self.kern.delta_11(self.Vu, self.Vu)
            Kinv, L = chol_inv(K11uu)
            self.register_buffer("K11uu", K11uu, persistent=False)
            self.register_buffer("K11uu_inv", Kinv, persistent=False)
            self.register_buffer("K11uu_chol", L, persistent=False)
            self.register_buffer("A", self.kern.delta_11(self.V, self.Vu) @ Kinv, persistent=False)
            self.register_buffer("B", self.kern.delta_21(self.V, self.Vu) @ Kinv, persistent=False)
            self.register_buffer("A2", self.A ** 2, persistent=False)
        self.logits = nn.Parameter(torch.randn(C, dtype=x.dtype, device=x.device))
        self.mus = nn.Parameter(torch.randn(C, self.m, dtype=x.dtype, device=x.device))
        self.s_params = nn.Parameter(torch.full((C, self.m), math.log(0.35), dtype=x.dtype, device=x.device))
        self.warm_start_mu()

    def _weights(self):
        return torch.softmax(self.logits, dim=0)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Load parameters only into the same inducing basis and kernel.

        Construct with checkpoint['Vu'] to reuse stochastic inducing locations.
        Data and base must also match: the large training caches are recomputed
        at construction, not stored in the checkpoint.
        """
        for name in ("Vu", "kern.lx2", "kern.lz2", "kern.sigma2"):
            if prefix + name not in state_dict:
                raise ValueError(f"Checkpoint is missing {name}; rebuild with the original configuration")
            current = self.Vu if name == "Vu" else getattr(self.kern, name.split('.')[1])
            saved = state_dict[prefix + name]
            if current.shape != saved.shape or not torch.equal(current.detach().cpu(), saved.detach().cpu()):
                raise ValueError(f"Checkpoint {name} differs; construct GPDR with checkpoint['Vu'] and the saved kernel configuration")
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                             missing_keys, unexpected_keys, error_msgs)

    def warm_start_mu(self):
        c = self.gy / self.g
        lin = 2.0 * self.gy + 2.0 * c * self.g
        W = self.g ** 2
        AtW = self.A.T * (W / self.n)
        H = self.K11uu_inv + 2.0 * self.beta * (AtW @ self.A)
        b = self.beta * (self.A.T @ (lin / self.n))
        if self.warm_start_solver == "solve":
            mu_ls = -torch.linalg.solve(H, b)
        else:
            try:
                L = torch.linalg.cholesky(H)
                mu_ls = -torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
            except RuntimeError:
                mu_ls = -torch.linalg.solve(H, b)
        with torch.no_grad():
            for k in range(self.C):
                self.mus[k].copy_(mu_ls / self.C + 0.03 * torch.randn_like(mu_ls))

    def forward_objective(self):
        w = self._weights()
        aMu = (self.A @ self.mus.T).T
        mu_bar = torch.sum(w[:, None] * self.mus, dim=0)
        Ez = self.A @ mu_bar
        if self.imputation == "gumbel":
            u = torch.rand_like(w).clamp_(1e-12, 1.0 - 1e-12)
            noise = -torch.log(-torch.log(u))
            pi_soft = torch.softmax((torch.log(w + 1e-12) + noise) / self.gumbel_tau, dim=0)
            eps = torch.randn_like(self.mus)
            samples = self.mus + torch.exp(self.s_params) * eps
            u_soft = torch.sum(pi_soft[:, None] * samples, dim=0)
            Efzz = self.B @ u_soft
        else:
            Efzz = self.B @ mu_bar
        g, gy, gyy = self.g, self.gy, self.gyy
        c = gy / g
        const = gyy / g - 0.5 * c ** 2
        lin_coeff = 2.0 * gy
        v_list = torch.exp(2.0 * self.s_params)
        row_var = (self.A2 @ v_list.T).T
        Efz2 = torch.sum(w[:, None] * (row_var + aMu ** 2), dim=0)
        EH = torch.mean(const + lin_coeff * Ez + 0.5 * g ** 2 * Efz2 + 1.0 * g ** 2 * Efzz)
        Kinv = self.K11uu_inv
        logdetK = (2.0 * torch.log(torch.diag(self.K11uu_chol)).sum()
                   if self.K11uu_chol is not None else torch.slogdet(self.K11uu)[1])
        diag_Kinv = torch.diag(Kinv)
        KL_k = []
        for k in range(self.C):
            mu_k, v_k = self.mus[k], v_list[k]
            quad = mu_k @ (Kinv @ mu_k)
            tr = torch.dot(diag_Kinv, v_k)
            logdetSigma = 2.0 * torch.sum(self.s_params[k])
            KL_k.append(0.5 * (tr + quad - self.m + logdetK - logdetSigma))
        KL = torch.sum(w * torch.stack(KL_k))
        if self.entropy_correction:
            KL = KL + torch.sum(w * torch.log(w + 1e-12))
        weighted = self.beta * self.n * EH if self.loss_reduction == "sum" else self.beta * EH
        return KL + weighted, {"KL": KL.detach(), "EH": EH.detach(),
                               "normalized EH": weighted.detach(), "w": w.detach()}

    @torch.no_grad()
    def correction(self, Vq):
        """Posterior-mean log correction at supplied (S(x), z) query rows."""
        K01 = self.kern.delta_01(Vq, self.Vu)
        m_tilde = torch.sum(self._weights()[:, None] * self.mus, dim=0)
        alpha = self.K11uu_inv @ m_tilde
        return K01 @ alpha

    _f_on_query = correction

    @torch.no_grad()
    def predict_f_and_derivs(self, x_star, y_min=-0.2, y_max=2.5, M=700, *, base=None):
        """Return (PIT, f, f_z, f_zz) on a response grid for any covariates.

        These are GP conditional means at the actual PIT, without the clipping
        used for density quadrature. This also supports inspecting a new base.
        """
        base = self.base_model if base is None else base
        xs = torch.as_tensor(x_star, dtype=self.x.dtype, device=self.x.device).reshape(1, -1)
        if xs.numel() != self.d_x:
            raise ValueError("x_star must have d covariates")
        y = torch.linspace(float(y_min), float(y_max), M, dtype=self.x.dtype, device=self.x.device)
        Xq = xs.repeat(M, 1)
        z = base.cdf(y, Xq[:, 0] if self.x.ndim == 1 else Xq)
        Vq = torch.cat([Xq, z.reshape(-1, 1)], dim=1)
        K01 = self.kern.delta_01(Vq, self.Vu)
        K11 = self.kern.delta_11(Vq, self.Vu)
        K21 = self.kern.delta_21(Vq, self.Vu)
        mu = torch.sum(self._weights()[:, None] * self.mus, dim=0)
        alpha = self.K11uu_inv @ mu
        return tuple(a.cpu().numpy() for a in (z, K01 @ alpha, K11 @ alpha, K21 @ alpha))

    @torch.no_grad()
    def predict_density(self, x_star, y_min=-0.2, y_max=2.5, M=700, *, base=None, reference_base=None):
        """Evaluate h on a y grid, normalizing exp(f) by trapezoids in PIT space.

        base may supply an already evaluated conditional base at this x (e.g.
        BetaBase(mu_star, phi)). reference_base is only a comparator: it never
        enters the GPDR density. Results have named fields, identical for every
        base family and covariate dimension.
        """
        if not isinstance(M, int) or M < 2 or not math.isfinite(float(y_min)) or not math.isfinite(float(y_max)) or float(y_min) >= float(y_max):
            raise ValueError("require M >= 2 and y_min < y_max")
        base = self.base_model if base is None else base
        reference_base = self.reference_base if reference_base is None else reference_base
        xs = torch.as_tensor(x_star, dtype=self.x.dtype, device=self.x.device).reshape(-1)
        if xs.numel() != self.d_x:
            raise ValueError("x_star must have d covariates")
        y = torch.linspace(float(y_min), float(y_max), M, dtype=self.x.dtype, device=self.x.device)
        Xq = xs.repeat(M, 1)
        base_x = Xq[:, 0] if self.x.ndim == 1 else Xq
        z = base.cdf(y, base_x).clamp(self.prediction_pit_epsilon, 1 - self.prediction_pit_epsilon)
        Vq = torch.cat([Xq, z.reshape(-1, 1)], dim=1)
        f = self.correction(Vq)
        g = base.pdf(y, base_x)
        g0 = g if reference_base is None else reference_base.pdf(y, base_x)
        idx = torch.argsort(z)
        zs, fs = z[idx], f[idx]
        dz = zs[1:] - zs[:-1]
        weights = torch.zeros_like(zs)
        weights[0], weights[-1] = dz[0] / 2.0, dz[-1] / 2.0
        weights[1:-1] = (dz[:-1] + dz[1:]) / 2.0
        exp_sorted = torch.exp(fs if self.exp_clip is None else torch.clamp(fs, -self.exp_clip, self.exp_clip))
        exp_f = torch.exp(f if self.exp_clip is None else torch.clamp(f, -self.exp_clip, self.exp_clip))
        denom = torch.sum(exp_sorted * weights) + 1e-12
        h = g * exp_f / denom
        return DensityPrediction(*(a.cpu().numpy() for a in (y, g, g0, h, exp_f, z)))


class LogisticGPMixtureDiag(GPDR):
    """Compatibility constructor; all mathematics is implemented by GPDR."""
    def __init__(self, x, y, *, base=None, C=2, m_induce=121, beta=1.0,
                 seed=2, kp=None, inducing="grid", scale_hyvarinen_by_n=False, **kwargs):
        def points(V):
            if inducing == "kmeans":
                return kmeans_inducing(V, m_induce)
            if inducing == "unit_grid":
                n = max(3, int(round(m_induce ** (1.0 / V.shape[1]))))
                return tensor_grid(*[torch.linspace(1e-6, 1 - 1e-6, n, dtype=V.dtype, device=V.device)
                                     for _ in range(V.shape[1])])
            if V.shape[1] == 2:
                n = int(math.sqrt(m_induce))
                return tensor_grid(torch.linspace(V[:, 0].min(), V[:, 0].max(), n, dtype=V.dtype, device=V.device),
                                   torch.linspace(1e-3, 1 - 1e-3, n, dtype=V.dtype, device=V.device))
            return sample_inducing(V, m_induce, seed)
        super().__init__(x, y, base=base or StudentTLinearBase(1.0, 1.0),
                         inducing_points=points, C=C, beta=beta, kp=kp,
                         normalization_size=x.numel(),
                         loss_reduction="sum" if scale_hyvarinen_by_n else "mean", **kwargs)


class BetaGPMixtureDiag(GPDR):
    """Compatibility constructor for a beta base; no separate GPDR algorithm."""
    def __init__(self, x, y, *, mu_base_train, phi_base, C=2, m_induce=200,
                 beta=50.0, seed=2, kp=None, **kwargs):
        super().__init__(x, y, base=BetaBase(mu_base_train, phi_base),
                         inducing_points=lambda V: sample_inducing(V, m_induce, seed),
                         C=C, beta=beta, kp=kp, train_pit_epsilon=1e-6, **kwargs)
