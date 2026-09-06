"""Native Python reproduction of the approved toy GPDR workflow.

Each function updates an explicit state mapping; no notebook is read at runtime.
"""
from pathlib import Path as FilePath
from IPython.display import display
from ..models import GPDR
from ..training import fit_adam
from .configuration import toy_model_inputs, TOY_FIT


def prepare(state):
    """Port of GPDR notebook cell 1 (one-based)."""
    import math
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import gpytorch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.distributions import Normal
    from dataclasses import dataclass
    from scipy.stats import norm, t as student_t
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2)
    np.random.seed(2)
    
    def make_toy(n=700, seed=3):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(n, generator=g)
        y = x + x ** 2 + torch.randn(n, generator=g) * (0.2 * x + 0.05)
        return (x, y)
    device = torch.device('cpu')
    n_train = state.get('n_train', 5000)
    n_test = state.get('n_test', 50000)
    (x_all, y_all) = make_toy(n=n_train + n_test, seed=state.get('seed_data', 3))
    rand_indices = torch.randperm(x_all.size(0), generator=torch.Generator().manual_seed(state.get('seed_split', 2)))
    (x, y) = (x_all[rand_indices[:n_train]], y_all[rand_indices[:n_train]])
    (x_test, y_test) = (x_all[rand_indices[n_train:]], y_all[rand_indices[n_train:]])
    x = x.to(device)
    y = y.to(device)
    x_np = x.cpu().numpy().reshape(-1, 1)
    y_np = y.cpu().numpy()
    slope = np.linalg.lstsq(x_np, y_np, rcond=None)[0][0]
    y_pred = slope * x_np.flatten()
    residuals = y_np - y_pred
    sig_base = np.std(residuals, ddof=1)
    b_base = slope
    SIGMA_NORMAL = sig_base * 3
    ORIGINAL_SIGMA_NORMAL = sig_base
    B_T = b_base
    SCALE_T = SIGMA_NORMAL
    DF = 3.0
    print(f'Base model g_tilde (Normal): loc = {b_base:.3f}*x, scale = {SIGMA_NORMAL:.3f}')
    print(f'Base model g (t-dist): loc = {B_T:.3f}*x, scale = {SCALE_T:.3f}, df = {DF}')
    
    def normal_pdf(y, mu, sigma):
        return 1.0 / (math.sqrt(2.0 * math.pi) * sigma) * torch.exp(-0.5 * ((y - mu) / sigma) ** 2)
    
    def t_pdf(y, mu, sigma, df=3.0):
        """
        Student's t-distribution PDF using scipy.
        Returns PDF of t(loc=mu, scale=sigma, df=df) at y.
        """
        mu_np = mu.cpu().numpy() if torch.is_tensor(mu) else mu
        sigma_np = sigma.cpu().numpy() if torch.is_tensor(sigma) else sigma
        y_np = y.cpu().numpy()
        pdf_np = student_t.pdf(y_np, df, loc=mu_np, scale=sigma_np)
        return torch.tensor(pdf_np, dtype=y.dtype, device=y.device)
    
    def base_g_terms(y, x, b=None, sigma=None, df=DF):
        """
        Base model using Student's t-distribution: t(loc=b*x, scale=sigma, df=df).
        Returns g(y|x), g'(y|x), g''(y|x).
        Uses scipy.stats.student_t.
        """
        if b is None:
            b = B_T
        if sigma is None:
            sigma = SCALE_T
        b = torch.as_tensor(b, dtype=x.dtype, device=x.device)
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        mu = b * x
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy() if torch.is_tensor(sigma) else sigma
        y_np = y.cpu().numpy()
        g_np = student_t.pdf(y_np, df, loc=mu_np, scale=sigma_np)
        g = torch.tensor(g_np, dtype=y.dtype, device=y.device)
        z = (y - mu) / sigma
        gy = -(df + 1) * z * g / (sigma * (df + z ** 2))
        gyy = g * (-(df + 1) / (sigma ** 2 * (df + z ** 2)) + (df + 1) * z ** 2 * (df + 3) / (sigma ** 2 * (df + z ** 2) ** 2))
        return (g, gy, gyy)
    
    def to_z_from_y(y, x, b=None, sigma=None, df=DF):
        """
        Transform y to z using CDF of Student's t-distribution.
        z = CDF_t((y - mu) / sigma)
        Uses t(loc=b*x, scale=sigma, df=df).
        """
        if b is None:
            b = B_T
        if sigma is None:
            sigma = SCALE_T
        b = torch.as_tensor(b, dtype=x.dtype, device=x.device)
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        mu = b * x
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
        y_np = y.cpu().numpy()
        z_np = student_t.cdf(y_np, df, loc=mu_np, scale=sigma_np)
        z = torch.tensor(z_np, dtype=y.dtype, device=y.device)
        return z
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def train(state):
    """Port of GPDR notebook cell 2 (one-based)."""
    B_T = state['B_T']
    DF = state['DF']
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    SCALE_T = state['SCALE_T']
    b_base = state['b_base']
    base_g_terms = state['base_g_terms']
    device = state['device']
    norm = state['norm']
    np = state['np']
    optim = state['optim']
    plt = state['plt']
    t_pdf = state['t_pdf']
    to_z_from_y = state['to_z_from_y']
    torch = state['torch']
    x = state['x']
    y = state['y']
    torch.manual_seed(2)
    np.random.seed(2)
    model = GPDR(**toy_model_inputs(state)).to(device)
    model, trace = fit_adam(model, **TOY_FIT)
    state.update(model=model, trace=trace)
    return training_plots(state)


def training_plots(state):
    """Render training summaries and densities from a publicly fitted model."""
    model, trace = state['model'], dict(state['trace'])
    if 's_component_mean' in trace:
        trace['s_mean'] = trace.pop('s_component_mean')
    np, plt, norm = state['np'], state['plt'], state['norm']
    b_base = state['b_base']
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    steps = range(len(trace['F']))
    (fig1, axes1) = plt.subplots(2, 2, figsize=(12, 8))
    axes1[0, 0].plot(steps, trace['F'], color='#2E86AB', linewidth=2)
    axes1[0, 0].set_xlabel('Closure calls')
    axes1[0, 0].set_ylabel('Objective F')
    axes1[0, 0].set_title('Objective Trace', fontweight='bold')
    axes1[0, 0].grid(True, alpha=0.3)
    W = np.stack(trace['w'])
    colors_weights = ['#A23B72', '#F18F01', '#6A0572', '#AB83A1', '#F08A5D', '#B83B5E', '#6A0572', '#C0E218', '#3F88C5', '#F2545B']
    for k in range(W.shape[1]):
        axes1[0, 1].plot(steps, W[:, k], label=f'ω_{k}', color=colors_weights[k], linewidth=2)
    axes1[0, 1].set_xlabel('Closure calls')
    axes1[0, 1].set_ylabel('Weight')
    axes1[0, 1].set_title('Mixture Weights Trace', fontweight='bold')
    axes1[0, 1].legend()
    axes1[0, 1].grid(True, alpha=0.3)
    MU = np.stack(trace['mu_norm'])
    for k in range(MU.shape[1]):
        axes1[1, 0].plot(steps, MU[:, k], label=f'||μ_{k}||', color=colors_weights[k], linewidth=2)
    axes1[1, 0].set_xlabel('Closure calls')
    axes1[1, 0].set_ylabel('Norm')
    axes1[1, 0].set_title('μ Norms Trace', fontweight='bold')
    axes1[1, 0].legend()
    axes1[1, 0].grid(True, alpha=0.3)
    S = np.stack(trace['s_mean'])
    for k in range(S.shape[1]):
        axes1[1, 1].plot(steps, S[:, k], label=f'mean(s_{k})', color=colors_weights[k], linewidth=2)
    axes1[1, 1].set_xlabel('Closure calls')
    axes1[1, 1].set_ylabel('Mean log-std')
    axes1[1, 1].set_title('s Means Trace', fontweight='bold')
    axes1[1, 1].legend()
    axes1[1, 1].grid(True, alpha=0.3)
    plt.suptitle('Optimization Traces', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('optimization_traces.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    (fig2, axes2) = plt.subplots(1, 3, figsize=(15, 5))
    x_values = [0.3, 0.5, 0.7]
    colors_density = ['blue', 'orange', 'green']
    for (i, (ax, x_star)) in enumerate(zip(axes2, x_values)):
        (y_grid, g_vals, _, h_hat, _, _) = model.predict_density(x_star, y_min=-0.3, y_max=3.0, M=2000)
        g_tilde_vals = norm.pdf(y_grid, loc=b_base * x_star, scale=ORIGINAL_SIGMA_NORMAL)
        ax.plot(y_grid, g_tilde_vals, '--', color=colors_density[i], linewidth=2.5, alpha=0.7, label='Base model $g_{\\mathrm{tilde}}$(y|x)')
        ax.plot(y_grid, h_hat, '-', color=colors_density[i], linewidth=3, label='Predicted ĥ(y|x)')
        ax.set_xlabel('y', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(y_grid.min(), y_grid.max())
        ax.fill_between(y_grid, 0, h_hat, alpha=0.2, color=colors_density[i])
    plt.tight_layout()
    plt.savefig('density_comparisons.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def panels(state):
    """Port of GPDR notebook cell 3 (one-based)."""
    B_T = state['B_T']
    DF = state['DF']
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    SCALE_T = state['SCALE_T']
    b_base = state['b_base']
    model = state['model']
    norm = state['norm']
    np = state['np']
    plt = state['plt']
    student_t = state['student_t']
    x = state['x']
    y = state['y']
    x_vals = [0.3, 0.5, 0.7]
    colors = ['blue', 'orange', 'green']
    (y_min, y_max, M) = (-0.3, 3.0, 2000)
    
    def compute_high_density_z_range(x_star, base_density_threshold=0.001, df=3.0):
        """
        Find z-range where true density h_true(y|x) is above threshold.
        Returns (z_min, z_max) corresponding to high-density y region.
        """
        mu_true = x_star + x_star ** 2
        sigma_true = 0.2 * x_star + 0.05
        y_lower = norm.ppf(0.0025, loc=mu_true, scale=sigma_true)
        y_upper = norm.ppf(0.9975, loc=mu_true, scale=sigma_true)
        z_lower = student_t.cdf(y_lower, df, loc=B_T * x_star, scale=SCALE_T)
        z_upper = student_t.cdf(y_upper, df, loc=B_T * x_star, scale=SCALE_T)
        z_lower = max(0.01, z_lower)
        z_upper = min(0.99, z_upper)
        return (z_lower, z_upper)
    
    def true_h_np(y_np, x_star):
        mu = x_star + x_star ** 2
        sd = 0.2 * x_star + 0.05
        return 1.0 / (np.sqrt(2 * np.pi) * sd) * np.exp(-0.5 * ((y_np - mu) / sd) ** 2)
    
    def f_val_visual(x, z, df=DF):
        y = student_t.ppf(z, df, loc=B_T * x, scale=SCALE_T)
        log_true = norm.logpdf(y, loc=x + x ** 2, scale=0.2 * x + 0.05)
        log_base = student_t.logpdf(y, df, loc=B_T * x, scale=SCALE_T)
        return log_true - log_base
    
    def f_z_visual(x, z, df=DF):
        y = student_t.ppf(z, df, loc=B_T * x, scale=SCALE_T)
        standardized = (y - B_T * x) / SCALE_T
        t_pdf_val = student_t.pdf(standardized, df, loc=0, scale=1)
        y_z = SCALE_T / t_pdf_val
        mu_true = x + x ** 2
        sigma_true = 0.2 * x + 0.05
        h_y_over_h_true = -(y - mu_true) / sigma_true ** 2
        g_y_over_g_base = -(df + 1.0) * standardized / (SCALE_T * (df + standardized ** 2))
        f_y = h_y_over_h_true - g_y_over_g_base
        return f_y * y_z
    
    def f_zz_visual(x, z, df=DF):
        y = student_t.ppf(z, df, loc=B_T * x, scale=SCALE_T)
        standardized = (y - B_T * x) / SCALE_T
        t_pdf_val = student_t.pdf(standardized, df, loc=0, scale=1)
        y_z = SCALE_T / t_pdf_val
        dt_dz = 1.0 / t_pdf_val
        dt_pdf_dt = t_pdf_val * (-(df + 1.0) * standardized / (df + standardized ** 2))
        y_zz = -SCALE_T * dt_pdf_dt * dt_dz / t_pdf_val ** 2
        mu_true = x + x ** 2
        sigma_true = 0.2 * x + 0.05
        h_yy_over_h_true = (y - mu_true) ** 2 / sigma_true ** 4 - 1.0 / sigma_true ** 2
        term1 = df + standardized ** 2
        g_yy_over_g_base = -(df + 1.0) / SCALE_T ** 2 * ((df - standardized ** 2) / term1 ** 2)
        f_yy = h_yy_over_h_true - g_yy_over_g_base
        h_y_over_h_true = -(y - mu_true) / sigma_true ** 2
        g_y_over_g_base = -(df + 1.0) * standardized / (SCALE_T * term1)
        f_y = h_y_over_h_true - g_y_over_g_base
        return f_yy * y_z ** 2 + f_y * y_zz
    
    def estimate_mean_quantiles(yg, hhat, level=0.95):
        dy = np.diff(yg)
        w = np.empty_like(hhat)
        w[0] = dy[0] / 2.0
        w[-1] = dy[-1] / 2.0
        w[1:-1] = (dy[:-1] + dy[1:]) / 2.0
        p = hhat * w
        p /= p.sum()
        mean = np.dot(yg, p)
        cdf = np.cumsum(p)
        lower_quantile = np.interp((1 - level) / 2, cdf, yg)
        upper_quantile = np.interp(1 - (1 - level) / 2, cdf, yg)
        return (mean, lower_quantile, upper_quantile)
    (fig_a, ax_a) = plt.subplots(1, 1, figsize=(7, 5))
    for (x_star, col) in zip(x_vals, colors):
        (yg, gg, _, hhat, _, _) = model.predict_density(x_star, y_min=y_min, y_max=y_max, M=M)
        (_, f, _, _) = model.predict_f_and_derivs(x_star, y_min=y_min, y_max=y_max, M=M)
        h_true = true_h_np(yg, x_star)
        g_tilde = norm.pdf(yg, loc=b_base * x_star, scale=ORIGINAL_SIGMA_NORMAL)
        ax_a.plot(yg, g_tilde, '--', color=col, lw=2, alpha=0.5, label=f'$g_0$(y|x), x={x_star}')
        ax_a.plot(yg, h_true, ':', color=col, lw=2, alpha=0.5, label=f'true h(y|x), x={x_star}')
        ax_a.plot(yg, hhat, '-', color=col, lw=3.5, label=f'$\\hat h(y|x)$, x={x_star}')
    ax_a.set_xlabel('y', fontsize=14)
    ax_a.set_ylabel('density', fontsize=14)
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(yg.min(), yg.max())
    plt.tight_layout()
    plt.savefig('Figure1_PanelA_densities.pdf', dpi=220, bbox_inches='tight')
    plt.show()
    (fig_b, ax_b) = plt.subplots(1, 1, figsize=(7, 5))
    for (x_star, col) in zip(x_vals, colors):
        (zq, f_model, fz_model, fzz_model) = model.predict_f_and_derivs(x_star, y_min=y_min, y_max=y_max, M=M)
        (z_min_dense, z_max_dense) = compute_high_density_z_range(x_star)
        mask = (zq > z_min_dense) & (zq < z_max_dense)
        z_plot = zq[mask]
        y_plot = student_t.ppf(z_plot, DF, loc=B_T * x_star, scale=SCALE_T)
        log_g = student_t.logpdf(y_plot, DF, loc=B_T * x_star, scale=SCALE_T)
        log_g_tilde = norm.logpdf(y_plot, loc=b_base * x_star, scale=ORIGINAL_SIGMA_NORMAL)
        log_g_over_g_tilde = log_g - log_g_tilde
        pred_log_f_over_gtilde = f_model[mask] + log_g_over_g_tilde
        log_h_true = norm.logpdf(y_plot, loc=x_star + x_star ** 2, scale=0.2 * x_star + 0.05)
        true_log_f_over_gtilde = log_h_true - log_g_tilde
        shift = np.mean(true_log_f_over_gtilde - pred_log_f_over_gtilde)
        pred_log_f_shifted = pred_log_f_over_gtilde + shift
        true_log_f_shifted = true_log_f_over_gtilde
        pred_ratio = np.exp(pred_log_f_shifted)
        true_ratio = np.exp(true_log_f_shifted)
        ax_b.plot(z_plot, pred_ratio, '-', color=col, lw=3.5, label='$\\exp(\\hat{f_0})$,' + f' x={x_star}')
        ax_b.plot(z_plot, true_ratio, ':', color=col, lw=2.5, alpha=0.7, label='true $\\exp(f_0)$,' + f' x={x_star}')
    ax_b.set_xlabel('G_0(y|x)', fontsize=14)
    ax_b.set_ylabel('$\\exp(f_0)$', fontsize=14)
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figure1_PanelB_density_ratio.pdf', dpi=220, bbox_inches='tight')
    plt.show()
    (fig_c, ax_c) = plt.subplots(1, 1, figsize=(7, 5))
    x_line = np.linspace(float(x.min()), float(x.max()), 500)
    mu_base = b_base * x_line
    fitted_predictions = []
    fitted_lower = []
    fitted_upper = []
    for x_star in x_line:
        (yg, gg, _, hhat, _, _) = model.predict_density(x_star, y_min=y_min, y_max=y_max, M=M)
        dy = yg[1] - yg[0]
        h_hat_norm = hhat / np.trapz(hhat, dx=dy)
        mean_y = np.trapz(yg * h_hat_norm, dx=dy)
        fitted_predictions.append(mean_y)
        cdf = np.cumsum(h_hat_norm) * dy
        cdf = np.clip(cdf, 0, 1)
        lower_quantile = np.interp(0.025, cdf, yg)
        upper_quantile = np.interp(0.975, cdf, yg)
        fitted_lower.append(lower_quantile)
        fitted_upper.append(upper_quantile)
    fitted_predictions = np.array(fitted_predictions)
    fitted_lower = np.array(fitted_lower)
    fitted_upper = np.array(fitted_upper)
    base_lower = mu_base + ORIGINAL_SIGMA_NORMAL * norm.ppf(0.025)
    base_upper = mu_base + ORIGINAL_SIGMA_NORMAL * norm.ppf(0.975)
    ax_c.scatter(x.cpu().numpy(), y.cpu().numpy(), s=6, alpha=0.5, color='k', label='data')
    x_pos = np.linspace(0, 1, 100)
    ax_c.plot(x_pos, x_pos + x_pos ** 2, color='black')
    ax_c.plot(x_pos, x_pos + x_pos ** 2 + 1.96 * (0.2 * x_pos + 0.05), color='black', linestyle='--', lw=3)
    ax_c.plot(x_pos, x_pos + x_pos ** 2 - 1.96 * (0.2 * x_pos + 0.05), color='black', linestyle='--', lw=3)
    ax_c.fill_between(x_line, base_lower, base_upper, color='lightblue', alpha=0.3, label='$g_0$ 95% CI (Normal)')
    ax_c.plot(x_line, mu_base, '--', color='lightblue', lw=2.5, alpha=0.7, label=f'$g_0$ mean ({b_base:.2f}x)')
    ci_mask = x_line >= 0.001
    ax_c.fill_between(x_line[ci_mask], fitted_lower[ci_mask], fitted_upper[ci_mask], color='red', alpha=0.5, label='fitted 95% CI')
    ax_c.plot(x_line, fitted_predictions, '-', color='red', lw=3, label='fitted model mean')
    ax_c.set_xlabel('x', fontsize=14)
    ax_c.set_ylabel('y', fontsize=14)
    ax_c.legend(loc='best', fontsize=12)
    ax_c.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figure1_PanelC_true_regression.pdf', dpi=220, bbox_inches='tight')
    plt.show()
    print('\nAll panels saved separately:')
    print('  - Figure1_PanelA_densities.pdf')
    print('  - Figure1_PanelB_density_ratio.pdf')
    print('  - Figure1_PanelC_true_regression.pdf')
    print('  - Figure1_PanelD_f_z_derivative.pdf')
    print('  - Figure1_PanelE_f_zz_derivative.pdf')
    import pickle
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def log_score(state):
    """Port of GPDR notebook cell 6 (one-based)."""
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    b_base = state['b_base']
    model = state['model']
    norm = state['norm']
    np = state['np']
    true_h_np = state['true_h_np']
    x_test = state['x_test']
    y_test = state['y_test']
    import time
    
    def format_elapsed_time(seconds):
        seconds = float(seconds)
        if seconds < 60:
            return f'{seconds:.2f}s'
        (minutes, sec) = divmod(seconds, 60)
        if minutes < 60:
            return f'{int(minutes)}m {sec:.0f}s'
        (hours, minutes) = divmod(minutes, 60)
        return f'{int(hours)}h {int(minutes)}m {sec:.0f}s'
    true_log_density_vals = 0
    estimated_log_density_vals = 0
    base_normal_log_density_vals = 0
    gpdr_logscore_start = time.perf_counter()
    for i in range(len(x_test)):
        x_star = float(x_test[i].cpu().numpy().item())
        y_star = float(y_test[i].cpu().numpy().item())
        (yg, gg, _, hhat, _, _) = model.predict_density(x_star, y_min=-0.3, y_max=3.0, M=2000)
        log_h_true_star = np.log(true_h_np(y_star, x_star))
        true_log_density_vals += log_h_true_star
        hhat_interp = np.interp(y_star, yg, hhat)
        log_hhat_star = np.log(hhat_interp)
        estimated_log_density_vals += log_hhat_star
        log_g_tilde_star = norm.logpdf(y_star, loc=b_base * x_star, scale=ORIGINAL_SIGMA_NORMAL)
        base_normal_log_density_vals += log_g_tilde_star
    gpdr_logscore_time_seconds = time.perf_counter() - gpdr_logscore_start
    print('=' * 70)
    print('MODEL COMPARISON: Log-likelihood on test set')
    print('=' * 70)
    print(f'1. True density h_true:                    {true_log_density_vals:.4f}')
    print(f'2. Estimated density ĥ (GP model):         {estimated_log_density_vals:.4f}')
    print(f'3. Normal-distributed base g_tilde:        {base_normal_log_density_vals:.4f}')
    print(f'GPDR log-score evaluation time: {format_elapsed_time(gpdr_logscore_time_seconds)}')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def metrics(state):
    """Port of GPDR notebook cell 7 (one-based)."""
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    b_base = state['b_base']
    format_elapsed_time = state['format_elapsed_time']
    model = state['model']
    norm = state['norm']
    np = state['np']
    time = state['time']
    x_test = state['x_test']
    y_max = state['y_max']
    y_min = state['y_min']
    y_test = state['y_test']
    M_eval = 1000
    n_test = len(x_test)
    ci_levels = [0.5, 0.8, 0.9, 0.95]
    gpdr_metric_start = time.perf_counter()
    updated_preds = []
    updated_intervals = {level: {'lower': [], 'upper': []} for level in ci_levels}
    normal_base_preds = []
    normal_base_intervals = {level: {'lower': [], 'upper': []} for level in ci_levels}
    y_obs = []
    for i in range(n_test):
        x_star = float(x_test[i].cpu().numpy().item())
        y_star = float(y_test[i].cpu().numpy().item())
        y_obs.append(y_star)
        (yg, gg, _, hhat, _, _) = model.predict_density(x_star, y_min=y_min, y_max=y_max, M=M_eval)
        dy = yg[1] - yg[0]
        h_hat_norm = hhat / np.trapz(hhat, dx=dy)
        mean_y = np.trapz(yg * h_hat_norm, dx=dy)
        updated_preds.append(mean_y)
        cdf = np.cumsum(h_hat_norm) * dy
        cdf = np.clip(cdf, 0, 1)
        for level in ci_levels:
            lower_q = (1 - level) / 2
            upper_q = 1 - lower_q
            lower_quantile = np.interp(lower_q, cdf, yg)
            upper_quantile = np.interp(upper_q, cdf, yg)
            updated_intervals[level]['lower'].append(lower_quantile)
            updated_intervals[level]['upper'].append(upper_quantile)
        mu_normal = b_base * x_star
        normal_base_preds.append(mu_normal)
        for level in ci_levels:
            lower_q = (1 - level) / 2
            upper_q = 1 - lower_q
            normal_base_intervals[level]['lower'].append(mu_normal + ORIGINAL_SIGMA_NORMAL * norm.ppf(lower_q))
            normal_base_intervals[level]['upper'].append(mu_normal + ORIGINAL_SIGMA_NORMAL * norm.ppf(upper_q))
    y_obs = np.array(y_obs)
    updated_preds = np.array(updated_preds)
    normal_base_preds = np.array(normal_base_preds)
    for level in ci_levels:
        updated_intervals[level]['lower'] = np.array(updated_intervals[level]['lower'])
        updated_intervals[level]['upper'] = np.array(updated_intervals[level]['upper'])
        normal_base_intervals[level]['lower'] = np.array(normal_base_intervals[level]['lower'])
        normal_base_intervals[level]['upper'] = np.array(normal_base_intervals[level]['upper'])
    rmse_updated = np.sqrt(np.mean((updated_preds - y_obs) ** 2))
    rmse_normal = np.sqrt(np.mean((normal_base_preds - y_obs) ** 2))
    print('PREDICTION PERFORMANCE METRICS')
    print(f'RMSE - Updated: {rmse_updated:.4f}  |  Normal-base: {rmse_normal:.4f}')
    print()
    for level in ci_levels:
        coverage_updated = np.mean((y_obs >= updated_intervals[level]['lower']) & (y_obs <= updated_intervals[level]['upper']))
        coverage_normal = np.mean((y_obs >= normal_base_intervals[level]['lower']) & (y_obs <= normal_base_intervals[level]['upper']))
        length_updated = np.mean(updated_intervals[level]['upper'] - updated_intervals[level]['lower'])
        length_normal = np.mean(normal_base_intervals[level]['upper'] - normal_base_intervals[level]['lower'])
        print(f'{int(level * 100)}% CI Coverage - Updated: {coverage_updated:.4f}  |  Normal-base: {coverage_normal:.4f}')
        print(f'{int(level * 100)}% CI Width   - Updated: {length_updated:.4f}  |  Normal-base: {length_normal:.4f}')
        print()
    gpdr_metric_time_seconds = time.perf_counter() - gpdr_metric_start
    gpdr_prediction_time_seconds = state.get('gpdr_logscore_time_seconds', 0.0) + gpdr_metric_time_seconds
    gpdr_prediction_time = format_elapsed_time(gpdr_prediction_time_seconds)
    print(f'GPDR prediction/evaluation time: {gpdr_prediction_time}')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def diagnostics(state):
    """Port of GPDR notebook cell 13 (one-based)."""
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    b_base = state['b_base']
    model = state['model']
    norm = state['norm']
    np = state['np']
    plt = state['plt']
    torch = state['torch']
    @torch.no_grad()
    def quantile_residual_diagnostic_toy(model, y_min=-0.3, y_max=3.0, M=1500, eps=1e-10):
        """Compute quantile residuals for GPDR and the normal base model g_0."""
        gpdr_pit = []
        gpdr_mean = []
        base_pit = []
        base_mean = []
        x_np = model.x.detach().cpu().numpy()
        y_np = model.y.detach().cpu().numpy()
        for (x_i, y_i) in zip(x_np, y_np):
            (y_grid, _, _, h_hat, _, _) = model.predict_density(float(x_i), y_min=y_min, y_max=y_max, M=M)
            dy = np.diff(y_grid)
            wts = np.zeros_like(y_grid)
            wts[0] = dy[0] / 2.0
            wts[-1] = dy[-1] / 2.0
            wts[1:-1] = (dy[:-1] + dy[1:]) / 2.0
            prob = h_hat * wts
            prob = prob / prob.sum()
            cdf = np.cumsum(prob)
            cdf = np.clip(cdf, 0.0, 1.0)
            pit_i = np.interp(y_i, y_grid, cdf)
            gpdr_pit.append(pit_i)
            gpdr_mean.append(float(np.sum(y_grid * prob)))
            mu_base_i = float(b_base * x_i)
            base_mean.append(mu_base_i)
            base_pit.append(norm.cdf(y_i, loc=mu_base_i, scale=ORIGINAL_SIGMA_NORMAL))
        gpdr_pit = np.clip(np.asarray(gpdr_pit), eps, 1.0 - eps)
        base_pit = np.clip(np.asarray(base_pit), eps, 1.0 - eps)
        gpdr_mean = np.asarray(gpdr_mean)
        base_mean = np.asarray(base_mean)
        gpdr_resid = norm.ppf(gpdr_pit)
        base_resid = norm.ppf(base_pit)
        return {'gpdr': {'pit': gpdr_pit, 'quantile_residual': gpdr_resid, 'fitted_mean': gpdr_mean, 'summary': {'mean': float(np.mean(gpdr_resid)), 'std': float(np.std(gpdr_resid, ddof=1)), 'q05': float(np.quantile(gpdr_resid, 0.05)), 'median': float(np.quantile(gpdr_resid, 0.5)), 'q95': float(np.quantile(gpdr_resid, 0.95))}}, 'base_g0': {'pit': base_pit, 'quantile_residual': base_resid, 'fitted_mean': base_mean, 'summary': {'mean': float(np.mean(base_resid)), 'std': float(np.std(base_resid, ddof=1)), 'q05': float(np.quantile(base_resid, 0.05)), 'median': float(np.quantile(base_resid, 0.5)), 'q95': float(np.quantile(base_resid, 0.95))}}}
    qr_diag = quantile_residual_diagnostic_toy(model)
    print('Toy quantile residual diagnostics')
    print('=' * 70)
    for (model_name, payload) in qr_diag.items():
        print()
        print(model_name)
        for (key, value) in payload['summary'].items():
            print(f'  {key:>6s}: {value:.6f}')
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    plot_specs = [('base_g0', axes[0], 'Base $g_0$ quantile residuals', '#457B9D'), ('gpdr', axes[1], 'GPDR quantile residuals', '#2A9D8F')]
    for (key, ax, title, color) in plot_specs:
        fitted_mean = qr_diag[key]['fitted_mean']
        residual = qr_diag[key]['quantile_residual']
        ax.scatter(fitted_mean, residual, s=10, alpha=0.35, color=color, edgecolors='none')
        ax.axhline(0.0, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('Fitted mean')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Quantile residual')
    plt.tight_layout()
    plt.savefig('toy_quantile_residuals_vs_fitted_mean.pdf', dpi=220, bbox_inches='tight')
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state

STAGES = (prepare, train, panels, log_score, metrics, diagnostics,)
