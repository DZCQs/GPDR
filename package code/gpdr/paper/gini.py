"""Native Python reproduction of the approved gini GPDR workflow.

Each function updates an explicit state mapping; no notebook is read at runtime.
"""
from pathlib import Path as FilePath
from IPython.display import display
from ..models import GPDR
from ..training import fit_adam
from .configuration import gini_model_inputs, GINI_FIT, predict_gini_row


def setup(state):
    """Port of GPDR notebook cell 2 (one-based)."""
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    from scipy.special import expit, gammaln
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 120)
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def load_data(state):
    """Port of GPDR notebook cell 4 (one-based)."""
    data_dir = state['data_dir']
    pd = state['pd']
    csv_path = data_dir / 'reg_df.csv'
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df.head()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def data_summary(state):
    """Port of GPDR notebook cell 6 (one-based)."""
    df = state['df']
    np = state['np']
    print('Rows, columns:', df.shape)
    print('\nMissing values per column:')
    display(df.isna().sum())
    y = df['y_adj'].to_numpy()
    print('\nResponse y_adj summary:')
    display(df['y_adj'].describe())
    print('All y_adj in (0,1):', bool(np.all((y > 0) & (y < 1))))
    display(df[['log_gdppc', 'urban', 'unemp', 'trade', 'year']].describe())
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def split(state):
    """Port of GPDR notebook cell 8 (one-based)."""
    df = state['df']
    np = state['np']
    cont_cols = ['log_gdppc', 'urban', 'unemp', 'trade', 'year']
    X_cont = df[cont_cols].copy()
    X = X_cont.copy()
    X.insert(0, 'Intercept', 1.0)
    y = df['y_adj'].to_numpy(dtype=float)
    X_mat = X.to_numpy(dtype=float)
    print('Design matrix shape:', X_mat.shape)
    print('Number of parameters in mean model (beta):', X_mat.shape[1])
    n = X_mat.shape[0]
    rng = np.random.default_rng(123)
    perm = rng.permutation(n)
    n_train = int(0.9 * n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    X_mat_train = X_mat[train_idx]
    X_mat_test = X_mat[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    print(f'Train size: {len(train_idx)}, Test size: {len(test_idx)}')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def fit_base(state):
    """Port of GPDR notebook cell 10 (one-based)."""
    X_mat_train = state['X_mat_train']
    expit = state['expit']
    gammaln = state['gammaln']
    minimize = state['minimize']
    np = state['np']
    y_train = state['y_train']
    def neg_loglik(params, y, X):
        """Negative log-likelihood for beta regression with logit mean and constant precision.
    
        params = [beta_0, ..., beta_{p-1}, log_phi]
        """
        beta = params[:-1]
        log_phi = params[-1]
        phi = np.exp(log_phi)
        eta = X @ beta
        mu = expit(eta)
        a = mu * phi
        b = (1 - mu) * phi
        ll = gammaln(phi) - gammaln(a) - gammaln(b) + (a - 1) * np.log(y) + (b - 1) * np.log(1 - y)
        return -np.sum(ll)
    p = X_mat_train.shape[1]
    init = np.zeros(p + 1)
    init[-1] = 0.0
    res = minimize(neg_loglik, init, args=(y_train, X_mat_train), method='BFGS', options={'disp': True, 'maxiter': 500})
    print('\nConverged:', res.success)
    print('Message:', res.message)
    print('Final negative log-likelihood (train):', res.fun)
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def coefficients(state):
    """Port of GPDR notebook cell 12 (one-based)."""
    X = state['X']
    np = state['np']
    pd = state['pd']
    res = state['res']
    beta_hat = res.x[:-1]
    log_phi_hat = res.x[-1]
    phi_hat = float(np.exp(log_phi_hat))
    Hinv = np.asarray(res.hess_inv)
    se = np.sqrt(np.diag(Hinv))
    param_names = list(X.columns) + ['log_phi']
    est = np.r_[beta_hat, log_phi_hat]
    out = pd.DataFrame({'param': param_names, 'estimate': est, 'std_err': se, 'z': est / se})
    phi_row = pd.DataFrame({'param': ['phi (precision)'], 'estimate': [phi_hat], 'std_err': [np.nan], 'z': [np.nan]})
    display(out.head(15))
    display(phi_row)
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def base_means(state):
    """Port of GPDR notebook cell 14 (one-based)."""
    X_mat_test = state['X_mat_test']
    X_mat_train = state['X_mat_train']
    beta_hat = state['beta_hat']
    df_test = state['df_test']
    df_train = state['df_train']
    expit = state['expit']
    np = state['np']
    mu_hat_train = expit(X_mat_train @ beta_hat)
    mu_hat_test = expit(X_mat_test @ beta_hat)
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train['mu_hat'] = mu_hat_train
    df_test['mu_hat'] = mu_hat_test
    display(df_test[['y_adj', 'mu_hat']].describe())
    err = df_test['y_adj'] - df_test['mu_hat']
    print('TEST MAE:', float(np.mean(np.abs(err))))
    print('TEST RMSE:', float(np.sqrt(np.mean(err ** 2))))
    display(df_test[['country', 'iso3c', 'year', 'y_adj', 'mu_hat', 'region', 'income']].head(10))
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def base_plots(state):
    """Port of GPDR notebook cell 16 (one-based)."""
    df_test = state['df_test']
    pd = state['pd']
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')
    assert 'mu_hat' in df_test.columns and 'y_adj' in df_test.columns, 'Expected columns missing in df_test'
    y_obs = df_test['y_adj'].to_numpy()
    y_hat = df_test['mu_hat'].to_numpy()
    resid = y_obs - y_hat
    (fig, axes) = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.scatter(y_hat, y_obs, s=12, alpha=0.6)
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='45° line')
    ax.set_xlabel('Predicted mean (mu_hat)')
    ax.set_ylabel('Observed (y_adj)')
    ax.set_title('Predicted vs Observed (TEST)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax = axes[0, 1]
    ax.scatter(y_hat, resid, s=12, alpha=0.6)
    ax.axhline(0.0, color='black', lw=1, linestyle='--')
    ax.set_xlabel('Predicted mean (mu_hat)')
    ax.set_ylabel('Residual (y - mu_hat)')
    ax.set_title('Residuals vs Fitted (TEST)')
    ax = axes[1, 0]
    bins = pd.qcut(y_hat, q=10, duplicates='drop')
    calib = df_test.assign(bin=bins).groupby('bin', observed=True).agg(mean_pred=('mu_hat', 'mean'), mean_obs=('y_adj', 'mean')).reset_index(drop=True)
    ax.plot(calib['mean_pred'], calib['mean_obs'], marker='o')
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    ax.set_xlabel('Avg predicted (by decile)')
    ax.set_ylabel('Avg observed (by decile)')
    ax.set_title('Calibration (Deciles, TEST)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax = axes[1, 1]
    sns.kdeplot(y_obs, ax=ax, label='Observed y (TEST)', fill=True, alpha=0.3, color='#1f77b4', clip=(0, 1))
    sns.kdeplot(y_hat, ax=ax, label='Predicted mu (TEST)', fill=True, alpha=0.3, color='#ff7f0e', clip=(0, 1))
    ax.set_xlabel('Value')
    ax.set_title('Distribution: Observed vs Predicted Mean (TEST)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def prepare(state):
    """Port of GPDR notebook cell 18 (one-based)."""
    df = state['df']
    df_test = state['df_test']
    df_train = state['df_train']
    mu_hat_train = state['mu_hat_train']
    phi_hat = state['phi_hat']
    import math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    np.random.seed(0)
    gp_cols = ['log_gdppc', 'urban', 'unemp', 'trade', 'year']
    X_gp_train = df_train[gp_cols].copy()
    X_gp_test = df_test[gp_cols].copy()
    X_gp_all = df[gp_cols].copy()
    x_min = X_gp_all.min(axis=0)
    x_max = X_gp_all.max(axis=0)
    X_gp_scaled = (X_gp_train - x_min) / (x_max - x_min + 1e-12)
    X_gp_scaled = X_gp_scaled.to_numpy(dtype=float)
    mu_base_train = mu_hat_train
    phi_base_original = phi_hat
    PHI_SCALE_FACTOR = 0.5
    phi_base = phi_base_original * PHI_SCALE_FACTOR
    print(f'Original precision phi: {phi_base_original:.4f}')
    print(f'Training precision phi (scaled by {PHI_SCALE_FACTOR}): {phi_base:.4f}')
    a_base = mu_base_train * phi_base
    b_base = (1.0 - mu_base_train) * phi_base
    eps = 1e-06
    y_np = np.clip(df_train['y_adj'].to_numpy(dtype=float), eps, 1.0 - eps)
    x_t = torch.from_numpy(X_gp_scaled)
    y_t = torch.from_numpy(y_np)
    a_t = torch.from_numpy(a_base)
    b_t = torch.from_numpy(b_base)
    
    def beta_pdf_torch(y, a, b):
        logB = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        logp = (a - 1.0) * torch.log(y) + (b - 1.0) * torch.log1p(-y) - logB
        return torch.exp(logp)
    
    def beta_cdf_torch(y, a, b):
        from scipy.special import betainc
        y_np = y.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        cdf_np = betainc(a_np, b_np, y_np)
        return torch.from_numpy(cdf_np).to(y.dtype).to(y.device)
    
    def beta_derivatives(y, a, b):
        g = beta_pdf_torch(y, a, b)
        dlog = (a - 1.0) / y - (b - 1.0) / (1.0 - y)
        d2log = -(a - 1.0) / y ** 2 - (b - 1.0) / (1.0 - y) ** 2
        gy = g * dlog
        gyy = g * (dlog ** 2 + d2log)
        return (g, gy, gyy)
    with torch.no_grad():
        (g_t, gy_t, gyy_t) = beta_derivatives(y_t, a_t, b_t)
        z_t = torch.clamp(beta_cdf_torch(y_t, a_t, b_t), 1e-06, 1 - 1e-06)
    V_t = torch.cat([x_t, z_t.reshape(-1, 1)], dim=1)
    (n, d) = x_t.shape
    print(f'GP TRAIN inputs shape V: {V_t.shape}  (n_train={n}, d_x={d})')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def train(state):
    """Port of GPDR notebook cell 21 (one-based)."""
    X_gp_scaled = state['X_gp_scaled']
    beta_hat = state['beta_hat']
    gp_cols = state['gp_cols']
    mu_base_train = state['mu_base_train']
    optim = state['optim']
    phi_base = state['phi_base']
    phi_base_original = state['phi_base_original']
    torch = state['torch']
    x_max = state['x_max']
    x_min = state['x_min']
    x_t = state['x_t']
    y_t = state['y_t']
    C = 2
    m_induce = 200
    beta_hyper = 493.61
    device = torch.device('cpu')
    x_dev = x_t.to(device)
    y_dev = y_t.to(device)
    model = GPDR(**gini_model_inputs(state)).to(device)
    model, trace = fit_adam(model, **GINI_FIT)
    trace = {key: trace[key] for key in ('F', 'w', 's_mean')}
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def density_plots(state):
    """Port of GPDR notebook cell 23 (one-based)."""
    X_gp_test = state['X_gp_test']
    X_test = state['X_test']
    df_test = state['df_test']
    df_train = state['df_train']
    model = state['model']
    mu_hat_test = state['mu_hat_test']
    np = state['np']
    pd = state['pd']
    plt = state['plt']
    trace = state['trace']
    import seaborn as sns
    sns.set(style='whitegrid')
    idx_sorted = np.argsort(mu_hat_test)
    sel_indices = [idx_sorted[int(0.01 * len(idx_sorted))], idx_sorted[int((0.01 + 0.91) / 2 * len(idx_sorted))], idx_sorted[int(0.91 * len(idx_sorted))]]
    sigma_gam = None
    try:
        if 'mu_hat_gam' in df_train.columns:
            resid_gam = df_train['y_adj'].to_numpy() - df_train['mu_hat_gam'].to_numpy()
            sigma_gam = float(np.std(resid_gam, ddof=1))
        elif 'gam' in locals():
            sigma_gam = float(np.sqrt(gam.statistics_.get('scale', np.nan)))
    except Exception:
        sigma_gam = None
    if sigma_gam is None or not np.isfinite(sigma_gam) or sigma_gam <= 0:
        sigma_gam = float(np.std(df_train['y_adj'].to_numpy(), ddof=1))
        if not np.isfinite(sigma_gam) or sigma_gam <= 0:
            sigma_gam = 0.001
    (fig2, axes2) = plt.subplots(1, 3, figsize=(15, 4))
    color_base = '#457B9D'
    color_updated = '#2A9D8F'
    color_obs = '#F77F00'
    for (j, (ax, ridx)) in enumerate(zip(axes2, sel_indices)):
        x_star = X_gp_test.iloc[ridx].to_dict()
        y_true = df_test['y_adj'].to_numpy()[ridx]
        X_row = X_test.iloc[ridx]
        (y_grid, _, g_vals, h_hat, exp_f, z_grid) = predict_gini_row(model, state, x_star_unscaled=x_star, X_row=X_row, df_row=df_test.iloc[ridx], y_min=0.001, y_max=1 - 0.001, M=1000)
        ax.plot(y_grid, g_vals, '--', color=color_base, linewidth=2.5, alpha=0.7, label=f'Base $g_0$(y|x)')
        ax.plot(y_grid, h_hat, '-', color=color_updated, linewidth=3.0, label='ĥ (updated)')
        ax.axvline(y_true, color=color_obs, linestyle=':', linewidth=2.5, alpha=0.9, label='Observed y (TEST)')
        ax.fill_between(y_grid, 0, h_hat, alpha=0.15, color=color_updated)
        row = df_test.iloc[ridx]
        country = row['country'] if 'country' in df_test.columns else ''
        year = row['year'] if 'year' in df_test.columns else ''
        try:
            year_str = str(int(year)) if year is not None and (not pd.isna(year)) else ''
        except Exception:
            year_str = str(year) if pd.notna(year) else ''
        title = f'{country} {year_str}'.strip() or f'Obs y = {y_true:.3f}'
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('y', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        if j == 0:
            ax.set_ylabel('Density', fontsize=14)
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Density Comparisons: Base g vs ĥ (updated) vs Observed', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig2.savefig('density_comparisons_base_updated_TEST.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def mc_means(state):
    """Port of GPDR notebook cell 24 (one-based)."""
    X_gp_test = state['X_gp_test']
    X_test = state['X_test']
    df_test = state['df_test']
    model = state['model']
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')
    if 'mu_hat_updated_mc' not in df_test.columns:
        N_eval = len(df_test)
        M_grid = 600
        S_samples = 1000
        rng = np.random.default_rng(123)
    
        def sample_from_density_grid(y_grid: np.ndarray, h_hat: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
            dy = np.diff(y_grid)
            wts = np.zeros_like(y_grid)
            wts[0] = dy[0] / 2.0
            wts[-1] = dy[-1] / 2.0
            wts[1:-1] = (dy[:-1] + dy[1:]) / 2.0
            prob = h_hat * wts
            prob_sum = prob.sum()
            if not np.isfinite(prob_sum) or prob_sum <= 0:
                prob = np.ones_like(prob) / prob.size
            else:
                prob = prob / prob_sum
            idx = rng.choice(y_grid.size, size=size, replace=True, p=prob)
            return y_grid[idx]
        updated_mean_mc = np.zeros(N_eval, dtype=float)
        for i in range(N_eval):
            x_star = X_gp_test.iloc[i].to_dict()
            X_row = X_test.iloc[i]
            (y_grid, _, _, h_hat, _, _) = predict_gini_row(model, state, x_star_unscaled=x_star, X_row=X_row, df_row=df_test.iloc[i], M=M_grid)
            ys = sample_from_density_grid(y_grid, h_hat, size=S_samples, rng=rng)
            updated_mean_mc[i] = float(ys.mean())
        df_test['mu_hat_updated_mc'] = updated_mean_mc
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def residual_plot(state):
    """Port of GPDR notebook cell 25 (one-based)."""
    X_mat_test = state['X_mat_test']
    beta_hat = state['beta_hat']
    df_test = state['df_test']
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')
    if 'mu_hat' not in df_test.columns:
        from scipy.special import expit
        mu_hat_test = expit(X_mat_test @ beta_hat)
        df_test['mu_hat'] = mu_hat_test
    if 'mu_hat_updated_mc' not in df_test.columns:
        raise RuntimeError('Run the MC updated-mean TEST cell first to populate mu_hat_updated_mc in df_test.')
    y_obs = df_test['y_adj'].to_numpy()
    y_hat_base = df_test['mu_hat'].to_numpy()
    y_hat_upd = df_test['mu_hat_updated_mc'].to_numpy()
    resid_base = y_obs - y_hat_base
    resid_upd = y_obs - y_hat_upd
    (fig, ax) = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(y_hat_upd, resid_upd, s=12, alpha=0.55, color='#2ca02c', label='Updated residuals (MC, TEST)')
    ax.scatter(y_hat_base, resid_base, s=12, alpha=0.55, color='#ff7f0e', label='Base residuals (TEST)')
    ax.axhline(0.0, color='black', lw=1, linestyle='--')
    ax.set_xlabel('Fitted μ̂ (base and updated, TEST)')
    ax.set_ylabel('Residual y − μ̂')
    ax.set_title('Residuals vs Fitted — Base vs Updated (MC, TEST)')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig('residuals_overlay_base_updated_mc_TEST.pdf', dpi=200, bbox_inches='tight')
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def log_score(state):
    """Port of GPDR notebook cell 26 (one-based)."""
    X_gp_test = state['X_gp_test']
    X_test = state['X_test']
    df_test = state['df_test']
    gammaln = state['gammaln']
    model = state['model']
    np = state['np']
    phi_hat = state['phi_hat']
    import time
    
    def format_elapsed_time(seconds):
        seconds = float(seconds)
        if seconds < 60:
            return f'{seconds:.2f}s'
        (minutes, seconds) = divmod(seconds, 60)
        if minutes < 60:
            return f'{int(minutes)}m {seconds:.1f}s'
        (hours, minutes) = divmod(minutes, 60)
        return f'{int(hours)}h {int(minutes)}m {seconds:.1f}s'
    gini_logscore_start = time.perf_counter()
    loglik_base = 0.0
    loglik_updated = 0.0
    for i in range(len(df_test)):
        y_star = float(df_test['y_adj'].iloc[i])
        mu_star = float(df_test['mu_hat'].iloc[i])
        a_star = mu_star * phi_hat
        b_star = (1.0 - mu_star) * phi_hat
        log_g = gammaln(a_star + b_star) - gammaln(a_star) - gammaln(b_star) + (a_star - 1.0) * np.log(y_star) + (b_star - 1.0) * np.log(1.0 - y_star)
        loglik_base += log_g
        x_star = X_gp_test.iloc[i].to_dict()
        X_row = X_test.iloc[i]
        (y_grid, _, g_grid, h_hat, _, _) = predict_gini_row(model, state, x_star_unscaled=x_star, X_row=X_row, df_row=df_test.iloc[i], y_min=0.001, y_max=1 - 0.001, M=1000)
        hhat_interp = np.interp(y_star, y_grid, h_hat)
        hhat_interp = max(hhat_interp, 1e-12)
        loglik_updated += np.log(hhat_interp)
    n_test = len(df_test)
    print(f'TEST log-likelihood (base Beta g):     {loglik_base:.4f}  (per obs: {loglik_base / n_test:.6f})')
    print(f'TEST log-likelihood (updated GP ĥ): {loglik_updated:.4f}  (per obs: {loglik_updated / n_test:.6f})')
    gini_logscore_time_seconds = time.perf_counter() - gini_logscore_start
    print(f'GPDR log-score evaluation time: {format_elapsed_time(gini_logscore_time_seconds)}')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def metrics(state):
    """Port of GPDR notebook cell 27 (one-based)."""
    X_gp_test = state['X_gp_test']
    X_test = state['X_test']
    beta_hat = state['beta_hat']
    df_test = state['df_test']
    expit = state['expit']
    format_elapsed_time = state['format_elapsed_time']
    model = state['model']
    phi_hat = state['phi_hat']
    import numpy as np
    from scipy.stats import beta as beta_dist
    import time
    gini_metric_start = time.perf_counter()
    n_samples = 3000
    n_test = len(df_test)
    
    def sample_from_base_model(X_row, n_samples):
        """Sample from base Beta(a, b) where a=mu*phi, b=(1-mu)*phi."""
        eta = beta_hat @ X_row.to_numpy()
        mu = expit(eta)
        a = mu * phi_hat
        b = (1.0 - mu) * phi_hat
        samples = beta_dist.rvs(a, b, size=n_samples, random_state=None)
        return samples
    
    def sample_from_updated_model(x_star_unscaled, X_row, df_row, n_samples, M=1000):
        """Sample from updated density h(y|x) via inverse transform sampling."""
        (y_grid, _, g_vals, h_hat, exp_f, z_grid) = predict_gini_row(model, state, x_star_unscaled=x_star_unscaled, X_row=X_row, df_row=df_row, y_min=0.001, y_max=1 - 0.001, M=M)
        dy = np.diff(y_grid)
        wts = np.zeros_like(y_grid)
        wts[0] = dy[0] / 2.0
        wts[-1] = dy[-1] / 2.0
        wts[1:-1] = (dy[:-1] + dy[1:]) / 2.0
        prob = h_hat * wts
        prob = prob / prob.sum()
        samples = np.random.choice(y_grid, size=n_samples, replace=True, p=prob)
        return samples
    base_samples_all = []
    updated_samples_all = []
    print('Generating samples for TEST set...')
    for i in range(n_test):
        if (i + 1) % 50 == 0:
            print(f'  Processing {i + 1}/{n_test}...')
        X_row = X_test.iloc[i]
        x_star = X_gp_test.iloc[i].to_dict()
        df_row = df_test.iloc[i]
        base_samp = sample_from_base_model(X_row, n_samples)
        base_samples_all.append(base_samp)
        updated_samp = sample_from_updated_model(x_star, X_row, df_row, n_samples)
        updated_samples_all.append(updated_samp)
    base_samples_all = np.array(base_samples_all)
    updated_samples_all = np.array(updated_samples_all)
    y_test_np = df_test['y_adj'].to_numpy()
    pred_base = base_samples_all.mean(axis=1)
    pred_updated = updated_samples_all.mean(axis=1)
    rmse_base = np.sqrt(np.mean((y_test_np - pred_base) ** 2))
    rmse_updated = np.sqrt(np.mean((y_test_np - pred_updated) ** 2))
    print('\n' + '=' * 60)
    print('Section 1: Root Mean Squared Error (RMSE)')
    print('=' * 60)
    print(f'Base model RMSE:    {rmse_base:.6f}')
    print(f'Updated model RMSE: {rmse_updated:.6f}')
    lower_base = np.percentile(base_samples_all, 2.5, axis=1)
    upper_base = np.percentile(base_samples_all, 97.5, axis=1)
    lower_updated = np.percentile(updated_samples_all, 2.5, axis=1)
    upper_updated = np.percentile(updated_samples_all, 97.5, axis=1)
    coverage_base = np.mean((y_test_np >= lower_base) & (y_test_np <= upper_base))
    coverage_updated = np.mean((y_test_np >= lower_updated) & (y_test_np <= upper_updated))
    print('\n' + '=' * 60)
    print('Section 2: 95% Credible Interval Coverage')
    print('=' * 60)
    print(f'Base model coverage:    {coverage_base:.4f}')
    print(f'Updated model coverage: {coverage_updated:.4f}')
    avg_length_base = np.mean(upper_base - lower_base)
    avg_length_updated = np.mean(upper_updated - lower_updated)
    print('\n' + '=' * 60)
    print('Section 3: Average 95% Credible Interval Length')
    print('=' * 60)
    print(f'Base model avg length:    {avg_length_base:.6f}')
    print(f'Updated model avg length: {avg_length_updated:.6f}')
    gini_metric_time_seconds = time.perf_counter() - gini_metric_start
    gpdr_prediction_time_seconds = state.get('gini_logscore_time_seconds', 0.0) + gini_metric_time_seconds
    gpdr_prediction_time = format_elapsed_time(gpdr_prediction_time_seconds)
    print(f'GPDR prediction/evaluation time: {gpdr_prediction_time}')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def diagnostics(state):
    """Port of GPDR notebook cell 34 (one-based)."""
    X_gp_test = state['X_gp_test']
    X_test = state['X_test']
    beta_dist = state['beta_dist']
    df_test = state['df_test']
    model = state['model']
    phi_hat = state['phi_hat']
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    def quantile_residual_diagnostic_gini(model, X_gp_eval, X_eval, df_eval, y_col='y_adj', y_min=0.001, y_max=1 - 0.001, M=1000, eps=1e-10, subset_size=None, seed=123):
        """Compute quantile residuals for GPDR and the Beta base model g_0, optionally on a reproducible subset."""
        gpdr_pit = []
        gpdr_mean = []
        base_pit = []
        base_mean = []
        n_total = len(df_eval)
        if subset_size is not None and subset_size < n_total:
            rng = np.random.default_rng(seed)
            use_idx = np.sort(rng.choice(n_total, size=subset_size, replace=False))
        else:
            use_idx = np.arange(n_total)
        X_gp_eval_use = X_gp_eval.iloc[use_idx].reset_index(drop=True)
        X_eval_use = X_eval.iloc[use_idx].reset_index(drop=True)
        df_eval_use = df_eval.iloc[use_idx].reset_index(drop=True)
        for i in range(len(df_eval_use)):
            y_obs = float(df_eval_use[y_col].iloc[i])
            x_star = X_gp_eval_use.iloc[i].to_dict()
            X_row = X_eval_use.iloc[i]
            (y_grid, _, _, h_hat, _, _) = predict_gini_row(model, state, x_star_unscaled=x_star, X_row=X_row, df_row=df_eval_use.iloc[i], y_min=y_min, y_max=y_max, M=M)
            dy = np.diff(y_grid)
            wts = np.zeros_like(y_grid)
            wts[0] = dy[0] / 2.0
            wts[-1] = dy[-1] / 2.0
            wts[1:-1] = (dy[:-1] + dy[1:]) / 2.0
            prob = h_hat * wts
            prob_sum = prob.sum()
            if not np.isfinite(prob_sum) or prob_sum <= 0:
                prob = np.ones_like(prob) / prob.size
            else:
                prob = prob / prob_sum
            cdf = np.cumsum(prob)
            cdf = np.clip(cdf, 0.0, 1.0)
            pit_i = np.interp(y_obs, y_grid, cdf)
            gpdr_pit.append(pit_i)
            gpdr_mean.append(float(np.sum(y_grid * prob)))
            mu_base_i = float(df_eval_use['mu_hat'].iloc[i])
            a_base_i = mu_base_i * phi_hat
            b_base_i = (1.0 - mu_base_i) * phi_hat
            base_mean.append(mu_base_i)
            base_pit.append(beta_dist.cdf(y_obs, a_base_i, b_base_i))
        gpdr_pit = np.clip(np.asarray(gpdr_pit), eps, 1.0 - eps)
        base_pit = np.clip(np.asarray(base_pit), eps, 1.0 - eps)
        gpdr_mean = np.asarray(gpdr_mean)
        base_mean = np.asarray(base_mean)
        gpdr_resid = norm.ppf(gpdr_pit)
        base_resid = norm.ppf(base_pit)
        return {'gpdr': {'pit': gpdr_pit, 'quantile_residual': gpdr_resid, 'fitted_mean': gpdr_mean, 'summary': {'mean': float(np.mean(gpdr_resid)), 'std': float(np.std(gpdr_resid, ddof=1)), 'q05': float(np.quantile(gpdr_resid, 0.05)), 'median': float(np.quantile(gpdr_resid, 0.5)), 'q95': float(np.quantile(gpdr_resid, 0.95))}}, 'base_g0': {'pit': base_pit, 'quantile_residual': base_resid, 'fitted_mean': base_mean, 'summary': {'mean': float(np.mean(base_resid)), 'std': float(np.std(base_resid, ddof=1)), 'q05': float(np.quantile(base_resid, 0.05)), 'median': float(np.quantile(base_resid, 0.5)), 'q95': float(np.quantile(base_resid, 0.95))}}, 'subset_idx': use_idx}
    gini_qr_diag = quantile_residual_diagnostic_gini(model, X_gp_test, X_test, df_test, y_col='y_adj', y_min=0.001, y_max=1 - 0.001, M=1000, subset_size=210, seed=123)
    print('Gini quantile residual diagnostics')
    print('=' * 60)
    print(f"subset_size_used: {len(gini_qr_diag['subset_idx'])}")
    for model_name in ['base_g0', 'gpdr']:
        payload = gini_qr_diag[model_name]
        print()
        print(model_name)
        for (key, value) in payload['summary'].items():
            print(f'  {key:>6s}: {value:.6f}')
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    plot_specs = [('base_g0', axes[0], 'Base $g_0$ quantile residuals', '#457B9D'), ('gpdr', axes[1], 'GPDR quantile residuals', '#2A9D8F')]
    all_resid = np.concatenate([gini_qr_diag['base_g0']['quantile_residual'], gini_qr_diag['gpdr']['quantile_residual']])
    y_lo = float(np.quantile(all_resid, 0.001))
    y_hi = float(np.quantile(all_resid, 0.999))
    rng = np.random.default_rng(123)
    x_min_common = float(min(gini_qr_diag['base_g0']['fitted_mean'].min(), gini_qr_diag['gpdr']['fitted_mean'].min())) - 0.02
    x_max_common = float(max(gini_qr_diag['base_g0']['fitted_mean'].max(), gini_qr_diag['gpdr']['fitted_mean'].max())) + 0.02
    for (key, ax, title, color) in plot_specs:
        fitted_mean = np.asarray(gini_qr_diag[key]['fitted_mean'])
        residual = np.asarray(gini_qr_diag[key]['quantile_residual'])
        keep_mask = np.ones(residual.shape[0], dtype=bool)
        if residual.shape[0] > 5:
            drop_idx = np.argsort(np.abs(residual))[-5:]
            keep_mask[drop_idx] = False
        fitted_mean_plot = fitted_mean[keep_mask]
        residual_plot = residual[keep_mask]
        jitter_scale = 0.004 * max(fitted_mean.max() - fitted_mean.min(), 1e-06)
        fitted_mean_jitter = fitted_mean_plot + rng.normal(0.0, jitter_scale, size=fitted_mean_plot.shape[0])
        ax.scatter(fitted_mean_jitter, residual_plot, s=18, alpha=0.28, color=color, edgecolors='none')
        ax.axhline(0.0, color='black', linestyle='--', linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel('Fitted mean')
        ax.set_xlim(x_min_common, x_max_common)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Quantile residual')
    plt.tight_layout()
    plt.savefig('gini_quantile_residuals_vs_fitted_mean.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state

STAGES = (setup, load_data, data_summary, split, fit_base, coefficients, base_means, base_plots, prepare, train, density_plots, mc_means, residual_plot, log_score, metrics, diagnostics,)
