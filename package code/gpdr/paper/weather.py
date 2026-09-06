"""Native Python reproduction of the approved weather GPDR workflow.

Each function updates an explicit state mapping; no notebook is read at runtime.
"""
from pathlib import Path as FilePath
from IPython.display import display
from ..models import GPDR
from ..training import fit_adam
from .configuration import weather_model_inputs, WEATHER_FIT


def setup(state):
    """Port of GPDR notebook cell 1 (one-based)."""
    import pandas as pd
    import numpy as np
    from pygam import LinearGAM, s, te, intercept
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from matplotlib.path import Path
    from shapely.geometry import Polygon, MultiPolygon
    from scipy.stats import norm, t as student_t
    from matplotlib import colors
    import torch.nn.functional as Fnn
    import math
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import gpytorch
    from torch.distributions import Normal
    from dataclasses import dataclass
    from kmeans_pytorch import kmeans
    torch.set_default_dtype(torch.float64)
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def seed(state):
    """Port of GPDR notebook cell 3 (one-based)."""
    np = state['np']
    torch = state['torch']
    torch.manual_seed(1)
    np.random.seed(1)
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def prepare(state):
    """Port of GPDR notebook cell 4 (one-based)."""
    data_dir = state['data_dir']
    np = state['np']
    pd = state['pd']
    df = pd.read_csv(data_dir / 'weather_data.csv', index_col='date')
    df.index = pd.to_datetime(df.index)
    df['t'] = (df.index - df.index[0]) / pd.Timedelta(days=1)
    df['t'] = df['t'] / df['t'].max()
    df = df[(df['t'] > 1 / 6) & (df['t'] < 2 / 6)].copy()
    df['t'] = (df['t'] - df['t'].min()) / (df['t'].max() - df['t'].min())
    X = df[['t', 'longitude', 'latitude']].values
    y = df['temperature'].values
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    X = (X - x_mean) / x_std
    X = 2 / (1 + np.exp(-X)) - 1
    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / y_std
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def split(state):
    """Port of GPDR notebook cell 5 (one-based)."""
    X = state['X']
    torch = state['torch']
    y = state['y']
    (x_all, y_all) = (X, y)
    rand_indices = torch.randperm(x_all.shape[0], generator=torch.Generator())
    (x_train, y_train) = (x_all[rand_indices[:-x_all.shape[0] // 10]], y_all[rand_indices[:-x_all.shape[0] // 10]])
    (x_test, y_test) = (x_all[rand_indices[-x_all.shape[0] // 10:]], y_all[rand_indices[-x_all.shape[0] // 10:]])
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def covariate_histogram(state):
    """Port of GPDR notebook cell 6 (one-based)."""
    plt = state['plt']
    x_all = state['x_all']
    plt.hist(x_all[:, 2], bins=50)
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def fit_base(state):
    """Port of GPDR notebook cell 7 (one-based)."""
    LinearGAM = state['LinearGAM']
    intercept = state['intercept']
    s = state['s']
    te = state['te']
    x_train = state['x_train']
    y_train = state['y_train']
    gam = LinearGAM(intercept + s(0, n_splines=5) + te(1, 2, n_splines=[5, 5]))
    gam.fit(x_train, y_train)
    print(gam.summary())
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def base_scale(state):
    """Port of GPDR notebook cell 8 (one-based)."""
    X = state['X']
    df = state['df']
    gam = state['gam']
    norm = state['norm']
    np = state['np']
    y = state['y']
    df['mu'] = gam.predict(X)
    df['sigma'] = np.std(y - df['mu'])
    df['z'] = norm(loc=df['mu'], scale=df['sigma']).cdf(y)
    SIGMA_NORMAL = np.std(y - df['mu'])
    ORIGINAL_SIGMA_NORMAL = SIGMA_NORMAL
    SIGMA_NORMAL = SIGMA_NORMAL * 3.0
    SCALE_T = SIGMA_NORMAL
    DF = 3.0
    print(f'Base model g_tilde (Normal GAM): scale = {SIGMA_NORMAL:.4f}')
    print(f'Base model g (t-dist): scale = {SCALE_T:.4f}, df = {DF}')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def pit_scatter(state):
    """Port of GPDR notebook cell 9 (one-based)."""
    DF = state['DF']
    SCALE_T = state['SCALE_T']
    df = state['df']
    plt = state['plt']
    student_t = state['student_t']
    y = state['y']
    df['z_t'] = student_t.cdf(y, DF, loc=df['mu'], scale=SCALE_T)
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(df.z, df.longitude, alpha=0.1, s=5)
    ax1.set_xlabel('z (from g_tilde, Normal)', fontsize=12)
    ax1.set_ylabel('Longitude', fontsize=12)
    ax1.set_title('z values for g_tilde (Normal GAM)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax2.scatter(df.z_t, df.longitude, alpha=0.1, s=5)
    ax2.set_xlabel('z (from g, t-dist)', fontsize=12)
    ax2.set_ylabel('Longitude', fontsize=12)
    ax2.set_title('z values for g (t-distribution)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def base_histogram(state):
    """Port of GPDR notebook cell 10 (one-based)."""
    df = state['df']
    plt = state['plt']
    plt.figure(figsize=(10, 5))
    plt.hist(df.z, bins=50, alpha=1, label='z from g_tilde (Normal)', color='blue', density=True)
    plt.hist(df.z_t, bins=50, alpha=1, label='z from g (t-dist)', color='orange', density=True)
    plt.xlabel('z value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Histogram of z values for g_tilde (Normal) vs g (t-dist)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def base_summary(state):
    """Port of GPDR notebook cell 11 (one-based)."""
    gam = state['gam']
    print(gam.distribution._known_scale)
    print(gam.statistics_['scale'])
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def map_setup(state):
    """Port of GPDR notebook cell 12 (one-based)."""
    import os
    os.environ['SHAPE_RESTORE_SHX'] = 'NO'
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def base_effects(state):
    """Port of GPDR notebook cell 13 (one-based)."""
    MultiPolygon = state['MultiPolygon']
    Path = state['Path']
    Polygon = state['Polygon']
    X = state['X']
    colors = state['colors']
    data_dir = state['data_dir']
    df = state['df']
    gam = state['gam']
    gpd = state['gpd']
    np = state['np']
    plt = state['plt']
    x_mean = state['x_mean']
    x_std = state['x_std']
    def points_in_polygon_path(xy, poly: Polygon):
        ext_path = Path(np.asarray(poly.exterior.coords))
        inside_ext = ext_path.contains_points(xy)
        if len(poly.interiors):
            inside_holes = np.zeros(xy.shape[0], dtype=bool)
            for ring in poly.interiors:
                hole_path = Path(np.asarray(ring.coords))
                inside_holes |= hole_path.contains_points(xy)
            return inside_ext & ~inside_holes
        else:
            return inside_ext
    f_t = gam.partial_dependence(term=1, X=X)
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, f_t, color='#2E86AB', linewidth=2)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature effect (standardized)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('base_model_temporal_effect.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    XX = gam.generate_X_grid(term=2, meshgrid=True)
    Z = gam.partial_dependence(term=2, X=XX, meshgrid=True)
    Z = Z - Z.mean()
    XX_original = [-np.log(2 / (XX[0] + 1) - 1) * x_std[1] + x_mean[1], -np.log(2 / (XX[1] + 1) - 1) * x_std[2] + x_mean[2]]
    gdf = gpd.read_file(data_dir / 'de.shp')
    gdf = gdf.explode(index_parts=False)
    x_flat = XX_original[0].ravel()
    y_flat = XX_original[1].ravel()
    xy = np.column_stack([x_flat, y_flat])
    outside = np.ones(xy.shape[0], dtype=bool)
    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            inside_poly = points_in_polygon_path(xy, geom)
            outside &= ~inside_poly
        elif isinstance(geom, MultiPolygon):
            inside_any = np.zeros(xy.shape[0], dtype=bool)
            for part in geom.geoms:
                inside_any |= points_in_polygon_path(xy, part)
            outside &= ~inside_any
        else:
            continue
    mask = outside.reshape(XX_original[0].shape)
    Z_masked = np.ma.array(Z, mask=mask)
    (fig, ax) = plt.subplots(figsize=(7, 6))
    vmin = np.nanmin(Z_masked)
    vmax = np.nanmax(Z_masked)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cm = ax.contourf(XX_original[0], XX_original[1], Z_masked, levels=300, cmap='coolwarm', norm=norm)
    gdf.boundary.plot(ax=ax, color='k', linewidth=1.0)
    fig.colorbar(cm, ax=ax, shrink=0.5, label='Temperature effect (standardized)')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('base_model_spatial_effect.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def helpers(state):
    """Port of GPDR notebook cell 15 (one-based)."""
    DF = state['DF']
    SCALE_T = state['SCALE_T']
    gam = state['gam']
    math = state['math']
    student_t = state['student_t']
    torch = state['torch']
    dim_x = 3
    
    def normal_pdf(y, mu, sigma):
        return 1.0 / (math.sqrt(2.0 * math.pi) * sigma) * torch.exp(-0.5 * ((y - mu) / sigma) ** 2)
    
    def t_pdf(y, mu, sigma, df=DF):
        """
        Student's t-distribution PDF using scipy.
        Returns PDF of t(loc=mu, scale=sigma, df=df) at y.
        """
        mu_np = mu.cpu().numpy() if torch.is_tensor(mu) else mu
        sigma_np = sigma if isinstance(sigma, float) else sigma.cpu().numpy() if torch.is_tensor(sigma) else sigma
        y_np = y.cpu().numpy()
        pdf_np = student_t.pdf(y_np, df, loc=mu_np, scale=sigma_np)
        return torch.tensor(pdf_np, dtype=y.dtype, device=y.device)
    
    def base_g_terms(y, x, sigma=SCALE_T, df=DF):
        """
        Base model using Student's t-distribution: t(loc=GAM(x), scale=sigma, df=df).
        Returns g(y|x), g'(y|x), g''(y|x).
        Uses scipy.stats.student_t.
        """
        mu = torch.tensor(gam.predict(x.cpu().numpy()), device=x.device, dtype=x.dtype)
        mu_np = mu.cpu().numpy()
        sigma_np = sigma if isinstance(sigma, float) else sigma
        y_np = y.cpu().numpy()
        g_np = student_t.pdf(y_np, df, loc=mu_np, scale=sigma_np)
        g = torch.tensor(g_np, dtype=y.dtype, device=y.device)
        z = (y - mu) / sigma
        gy = -(df + 1) * z * g / (sigma * (df + z ** 2))
        gyy = g * (-(df + 1) / (sigma ** 2 * (df + z ** 2)) + (df + 1) * z ** 2 * (df + 3) / (sigma ** 2 * (df + z ** 2) ** 2))
        return (g, gy, gyy)
    
    def to_z_from_y(y, x, sigma=SCALE_T, df=DF):
        """
        Transform y to z using CDF of Student's t-distribution.
        z = CDF_t((y - mu) / sigma)
        Uses t(loc=GAM(x), scale=sigma, df=df).
        """
        mu = torch.tensor(gam.predict(x.cpu().numpy()), device=x.device, dtype=x.dtype)
        mu_np = mu.cpu().numpy()
        sigma_np = sigma if isinstance(sigma, float) else sigma
        y_np = y.cpu().numpy()
        z_np = student_t.cdf(y_np, df, loc=mu_np, scale=sigma_np)
        z = torch.tensor(z_np, dtype=y.dtype, device=y.device)
        return z
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def train(state):
    """Port of GPDR notebook cell 16 (one-based)."""
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    base_g_terms = state['base_g_terms']
    gam = state['gam']
    optim = state['optim']
    to_z_from_y = state['to_z_from_y']
    torch = state['torch']
    x_test = state['x_test']
    x_train = state['x_train']
    y_test = state['y_test']
    y_train = state['y_train']
    number_of_steps = 5000
    step_verbose = 500
    device = torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    if device.type != 'cpu':
        torch.set_default_dtype(torch.float32)
    (x_train, y_train) = (torch.tensor(x_train, device=device, dtype=torch.float32), torch.tensor(y_train, device=device, dtype=torch.float32))
    (x_test, y_test) = (torch.tensor(x_test, device=device, dtype=torch.float32), torch.tensor(y_test, device=device, dtype=torch.float32))
    if device.type != 'cpu':
        x_test = x_test.to(device, dtype=torch.float32)
        y_test = y_test.to(device, dtype=torch.float32)
        x_train = x_train.to(device, dtype=torch.float32)
        y_train = y_train.to(device, dtype=torch.float32)
    else:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        x_train = x_train.to(device)
        y_train = y_train.to(device)
    import io
    import sys
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = GPDR(**weather_model_inputs(dict(state, x_train=x_train, y_train=y_train))).to(device)
    model, trace = fit_adam(model, **WEATHER_FIT)
    trace = {key: trace[key] for key in ('F', 'w', 'mu_norm', 's_component_mean')}
    trace['s_mean'] = trace.pop('s_component_mean')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def density_plots(state):
    """Port of GPDR notebook cell 17 (one-based)."""
    df = state['df']
    model = state['model']
    np = state['np']
    plt = state['plt']
    torch = state['torch']
    trace = state['trace']
    x_mean = state['x_mean']
    x_std = state['x_std']
    x_test = state['x_test']
    y_all = state['y_all']
    y_mean = state['y_mean']
    y_std = state['y_std']
    y_test = state['y_test']
    (fig1, axes1) = plt.subplots(2, 2, figsize=(12, 8))
    axes1[0, 0].plot(range(1, len(trace['F']) + 1), trace['F'], color='#2E86AB', linewidth=2)
    axes1[0, 0].set_xlabel('Closure calls')
    axes1[0, 0].set_ylabel('Objective F')
    axes1[0, 0].set_title('Objective Trace', fontweight='bold')
    axes1[0, 0].grid(True, alpha=0.3)
    W = np.stack(trace['w'])
    colors_weights = ['#A23B72', '#F18F01', '#6A0572', '#AB83A1', '#F08A5D', '#B83B5E', '#6A0572', '#C0E218', '#3F88C5', '#F2545B']
    for k in range(W.shape[1]):
        axes1[0, 1].plot(range(1, len(trace['F']) + 1), W[:, k], label=f'ω_{k}', color=colors_weights[k], linewidth=2)
    axes1[0, 1].set_xlabel('Closure calls')
    axes1[0, 1].set_ylabel('Weight')
    axes1[0, 1].set_title('Mixture Weights Trace', fontweight='bold')
    axes1[0, 1].legend()
    axes1[0, 1].grid(True, alpha=0.3)
    MU = np.stack(trace['mu_norm'])
    for k in range(MU.shape[1]):
        axes1[1, 0].plot(range(1, len(trace['F']) + 1), MU[:, k], label=f'||μ_{k}||', color=colors_weights[k], linewidth=2)
    axes1[1, 0].set_xlabel('Closure calls')
    axes1[1, 0].set_ylabel('Norm')
    axes1[1, 0].set_title('μ Norms Trace', fontweight='bold')
    axes1[1, 0].legend()
    axes1[1, 0].grid(True, alpha=0.3)
    S = np.stack(trace['s_mean'])
    for k in range(S.shape[1]):
        axes1[1, 1].plot(range(1, len(trace['F']) + 1), S[:, k], label=f'mean(s_{k})', color=colors_weights[k], linewidth=2)
    axes1[1, 1].set_xlabel('Closure calls')
    axes1[1, 1].set_ylabel('Mean log-std')
    axes1[1, 1].set_title('s Means Trace', fontweight='bold')
    axes1[1, 1].legend()
    axes1[1, 1].grid(True, alpha=0.3)
    plt.suptitle('Optimization Traces', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('optimization_traces_weather.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    (fig2, axes2) = plt.subplots(1, 3, figsize=(15, 5))
    random_indexes = torch.randperm(x_test.size(0), generator=torch.Generator())[[12, 123, 1234]]
    x_values = [x_test[random_indexes[0], :].cpu().numpy(), x_test[random_indexes[1], :].cpu().numpy(), x_test[random_indexes[2], :].cpu().numpy()]
    y_values = [y_test[random_indexes[0]].cpu().numpy(), y_test[random_indexes[1]].cpu().numpy(), y_test[random_indexes[2]].cpu().numpy()]
    color_g = '#E63946'
    color_g_tilde = '#457B9D'
    color_h_hat = '#2A9D8F'
    color_obs = '#F77F00'
    for (i, (ax, x_star)) in enumerate(zip(axes2, x_values)):
        (y_grid, g_vals, g_tilde_vals, h_hat, _, _) = model.predict_density(x_star, y_all.min(), y_all.max(), M=1000)
        x_star_original = -np.log(2 / (x_star + 1) - 1) * x_std + x_mean
        y_grid_original = y_grid * y_std + y_mean
        y_observed_original = y_values[i] * y_std + y_mean
        t_normalized = x_star_original[0]
        date_range_days = (df.index.max() - df.index.min()).days
        day_number = int(t_normalized * date_range_days)
        ax.plot(y_grid_original, g_tilde_vals, '--', color=color_g_tilde, linewidth=2.5, alpha=0.7, label=f'$g_0$ (Normal GAM)')
        ax.plot(y_grid_original, h_hat, '-', color=color_h_hat, linewidth=3, label='ĥ (predicted)')
        ax.axvline(y_observed_original, color=color_obs, linestyle=':', linewidth=2.5, alpha=0.8, label='Observed y')
        ax.set_xlabel('Temperature (°C)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(f'Location {i + 1}: Day {day_number}\nLon={x_star_original[1]:.2f}, Lat={x_star_original[2]:.2f}', fontsize=12)
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(y_grid_original.min(), y_grid_original.max())
        ax.fill_between(y_grid_original, 0, h_hat, alpha=0.15, color=color_h_hat)
    plt.suptitle(f'Density Comparisons: $g_0$ (Normal-base) vs ĥ (predicted) vs Observed', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig('density_comparisons_weather_gpu.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def log_score(state):
    """Port of GPDR notebook cell 18 (one-based)."""
    model = state['model']
    np = state['np']
    x_test = state['x_test']
    y_all = state['y_all']
    y_test = state['y_test']
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
    weather_logscore_start = time.perf_counter()
    estimated_log_density_vals = 0
    base_tilde_log_density_vals = 0
    for i in range(len(x_test)):
        x_star = x_test[i].cpu().numpy()
        y_star = float(y_test[i].cpu().numpy().item())
        (yg, gg, gg_tilde, hhat, _, _) = model.predict_density(x_star, y_min=y_all.min(), y_max=y_all.max(), M=1000)
        hhat_interp = np.interp(y_star, yg, hhat)
        log_hhat_star = np.log(hhat_interp)
        estimated_log_density_vals += log_hhat_star
        g_tilde_interp = np.interp(y_star, yg, gg_tilde)
        log_g_tilde_star = np.log(g_tilde_interp)
        base_tilde_log_density_vals += log_g_tilde_star
    print(f'Estimated Log Density (ĥ): {estimated_log_density_vals:.4f}')
    print(f'Base Model Log Density ($g_0$, Normal): {base_tilde_log_density_vals:.4f}')
    weather_logscore_time_seconds = time.perf_counter() - weather_logscore_start
    print(f'GPDR log-score evaluation time: {format_elapsed_time(weather_logscore_time_seconds)}')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def prediction_helpers(state):
    """Port of GPDR notebook cell 20 (one-based)."""
    DF = state['DF']
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    SCALE_T = state['SCALE_T']
    gam = state['gam']
    model = state['model']
    np = state['np']
    student_t = state['student_t']
    def get_base_g_stats(x_star):
        """
        Get mean and 95% credible interval for base model g (t-distribution).
        Uses scipy.stats.student_t.ppf for quantiles.
        
        Parameters:
        - x_star: array-like, shape (3,), containing [t, lon, lat] (standardized)
        
        Returns:
        - mu, lower, upper: mean and 95% CI bounds
        """
        x_star = np.array(x_star).reshape(1, -1)
        mu = gam.predict(x_star)[0]
        lower = student_t.ppf(0.025, DF, loc=mu, scale=SCALE_T)
        upper = student_t.ppf(0.975, DF, loc=mu, scale=SCALE_T)
        return (mu, lower, upper)
    
    def get_base_g_tilde_stats(x_star):
        """
        Get mean and 95% credible interval for base model g_tilde (Normal GAM).
        Uses scipy.stats.norm.ppf for quantiles with ORIGINAL_SIGMA_NORMAL (unscaled).
        
        Parameters:
        - x_star: array-like, shape (3,), containing [t, lon, lat] (standardized)
        
        Returns:
        - mu, lower, upper: mean and 95% CI bounds
        """
        from scipy.stats import norm
        x_star = np.array(x_star).reshape(1, -1)
        mu = gam.predict(x_star)[0]
        lower = norm.ppf(0.025, loc=mu, scale=ORIGINAL_SIGMA_NORMAL)
        upper = norm.ppf(0.975, loc=mu, scale=ORIGINAL_SIGMA_NORMAL)
        return (mu, lower, upper)
    
    def sample_from_base_model(x_star, n_samples=1000):
        """
        Generate samples from the Normal GAM base model g_tilde at a given location/time.
        Uses ORIGINAL_SIGMA_NORMAL (unscaled).
        
        Parameters:
        - x_star: array-like, shape (3,), containing [t, lon, lat] (standardized)
        - n_samples: int, number of samples to generate
        
        Returns:
        - samples: array of sampled temperatures
        """
        x_star = np.array(x_star).reshape(1, -1)
        mu = gam.predict(x_star)[0]
        samples = np.random.normal(loc=mu, scale=ORIGINAL_SIGMA_NORMAL, size=n_samples)
        return samples
    
    def sample_from_updated_model(x_star, n_samples=1000, y_min=None, y_max=None, M=500):
        """
        Generate samples (Inverse transform sampling) from the updated GP density model at a given location/time.
        
        Parameters:
        - x_star: array-like, shape (3,), containing [t, lon, lat]
        - n_samples: int, number of samples to generate
        - y_min, y_max: float, range for temperature grid (if None, use reasonable defaults)
        - M: int, number of grid points for density evaluation
        
        Returns:
        - samples: array of sampled temperatures
        """
        x_star = np.array(x_star)
        (y_grid, g_vals, g_tilde_vals, h_hat, _, _) = model.predict_density(x_star, y_min=y_min, y_max=y_max, M=M)
        dy = y_grid[1] - y_grid[0]
        h_hat_norm = h_hat / np.trapz(h_hat, dx=dy)
        cdf = np.cumsum(h_hat_norm) * dy
        cdf = np.clip(cdf, 0, 1)
        np.random.seed(42)
        u = np.random.uniform(0, 1, n_samples)
        samples = np.interp(u, cdf, y_grid)
        return samples
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def time_effects(state):
    """Port of GPDR notebook cell 21 (one-based)."""
    df = state['df']
    get_base_g_tilde_stats = state['get_base_g_tilde_stats']
    model = state['model']
    np = state['np']
    plt = state['plt']
    x_mean = state['x_mean']
    x_std = state['x_std']
    y_all = state['y_all']
    y_mean = state['y_mean']
    y_std = state['y_std']
    from scipy.stats import norm
    for location_idx in [15555, 45563]:
        selected_location = df.iloc[location_idx]
        selected_lon = selected_location['longitude']
        selected_lat = selected_location['latitude']
        location_mask = (np.abs(df['longitude'] - selected_lon) < 0.01) & (np.abs(df['latitude'] - selected_lat) < 0.01)
        location_data = df[location_mask].copy()
        if len(location_data) < 10:
            distances = np.sqrt((df['longitude'] - selected_lon) ** 2 + (df['latitude'] - selected_lat) ** 2)
            closest_indices = np.argsort(distances)[:100]
            location_data = df.iloc[closest_indices].copy()
        location_data = location_data.sort_values('t')
        X_loc_original = location_data[['t', 'longitude', 'latitude']].values
        y_loc_original = location_data['temperature'].values
        X_loc = (X_loc_original - x_mean) / x_std
        X_loc = 2 / (1 + np.exp(-X_loc)) - 1
        y_loc = (y_loc_original - y_mean) / y_std
        mu_g_tilde = []
        lower_g_tilde = []
        upper_g_tilde = []
        mu_updated = []
        lower_updated = []
        upper_updated = []
        for (i, (t_val, lon_val, lat_val)) in enumerate(X_loc):
            x_star = np.array([t_val, lon_val, lat_val])
            (mu, lower, upper) = get_base_g_tilde_stats(x_star)
            mu_g_tilde.append(mu)
            lower_g_tilde.append(lower)
            upper_g_tilde.append(upper)
            try:
                (yg, gg, gg_tilde, hhat, _, _) = model.predict_density(x_star, y_min=y_all.min().item() - 20, y_max=y_all.max().item() + 20, M=1000)
                dy = yg[1] - yg[0]
                h_hat_norm = hhat / np.trapz(hhat, dx=dy)
                mean_y = np.trapz(yg * h_hat_norm, dx=dy)
                mu_updated.append(mean_y)
                cdf = np.cumsum(h_hat_norm) * dy
                cdf = np.clip(cdf, 0, 1)
                lower_quantile = np.interp(0.025, cdf, yg)
                upper_quantile = np.interp(0.975, cdf, yg)
                lower_updated.append(lower_quantile)
                upper_updated.append(upper_quantile)
            except:
                mu_updated.append(mu_g_tilde[-1])
                lower_updated.append(lower_g_tilde[-1])
                upper_updated.append(upper_g_tilde[-1])
        mu_g_tilde = np.array(mu_g_tilde)
        lower_g_tilde = np.array(lower_g_tilde)
        upper_g_tilde = np.array(upper_g_tilde)
        mu_updated = np.array(mu_updated)
        lower_updated = np.array(lower_updated)
        upper_updated = np.array(upper_updated)
        mu_g_tilde_original = mu_g_tilde * y_std + y_mean
        lower_g_tilde_original = lower_g_tilde * y_std + y_mean
        upper_g_tilde_original = upper_g_tilde * y_std + y_mean
        mu_updated_original = mu_updated * y_std + y_mean
        lower_updated_original = lower_updated * y_std + y_mean
        upper_updated_original = upper_updated * y_std + y_mean
        (fig, ax) = plt.subplots(1, 1, figsize=(15, 8))
        ax.scatter(location_data.index, y_loc_original, alpha=0.7, s=30, color='black', label='Observed Temperature', zorder=5)
        ax.plot(location_data.index, mu_g_tilde_original, '--', linewidth=2.5, color='#FF6B35', label=f'$g_0$ (Normal GAM)', alpha=0.8)
        ax.fill_between(location_data.index, lower_g_tilde_original, upper_g_tilde_original, alpha=0.2, color='#FF6B35', label=f'$g_0$ 95% CI')
        ax.plot(location_data.index, mu_updated_original, '-', linewidth=2.5, color='#004E89', label='ĥ (GPDR)', alpha=0.8)
        ax.fill_between(location_data.index, lower_updated_original, upper_updated_original, alpha=0.2, color='#004E89', label='ĥ 95% CI')
        ax.set_xlabel('Date', fontsize=16)
        ax.set_ylabel('Temperature (°C)', fontsize=16)
        ax.set_title(f'Location {location_idx}: ({selected_lon:.2f}°, {selected_lat:.2f}°)', fontsize=14)
        ax.legend(fontsize=11, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        plt.savefig(f'time_effect_weather_gpu_{location_idx}.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def spatial_effects(state):
    """Port of GPDR notebook cell 22 (one-based)."""
    MultiPolygon = state['MultiPolygon']
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    Polygon = state['Polygon']
    df = state['df']
    gam = state['gam']
    gdf = state['gdf']
    np = state['np']
    plt = state['plt']
    points_in_polygon_path = state['points_in_polygon_path']
    sample_from_updated_model = state['sample_from_updated_model']
    x_mean = state['x_mean']
    x_std = state['x_std']
    y_all = state['y_all']
    y_mean = state['y_mean']
    y_std = state['y_std']
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    index_day = 51
    fixed_time = df['t'].unique()[index_day - 1]
    obs_data = df[df['t'] == fixed_time].copy()
    lons_all = obs_data['longitude'].values
    lats_all = obs_data['latitude'].values
    temps_obs_all = obs_data['temperature'].values
    lons = lons_all
    lats = lats_all
    temps_obs = temps_obs_all
    print(f'Observed data points at fixed time (day={index_day}th): {len(temps_obs)} locations')
    print(f'Observed temperature range: [{temps_obs.min():.1f}, {temps_obs.max():.1f}]°C, mean: {temps_obs.mean():.1f}°C')
    print(f'Below 0°C count: {(temps_obs < 0).sum()} | Above 0°C count: {(temps_obs >= 0).sum()}')
    X_fixed_time_original = np.column_stack([np.full(len(lons), fixed_time), lons, lats])
    X_fixed_time = (X_fixed_time_original - x_mean) / x_std
    X_fixed_time = 2 / (1 + np.exp(-X_fixed_time)) - 1
    mu_g_tilde = gam.predict(X_fixed_time)
    sigma_g_tilde = ORIGINAL_SIGMA_NORMAL
    mu_g_tilde_original = mu_g_tilde * y_std + y_mean
    sigma_g_tilde_original = sigma_g_tilde * y_std
    print(f'g (Normal GAM) predictions: [{mu_g_tilde_original.min():.1f}, {mu_g_tilde_original.max():.1f}]°C, mean: {mu_g_tilde_original.mean():.1f}°C')
    print(f'g uncertainty: {sigma_g_tilde_original:.2f}°C (constant)')
    print('Computing ĥ (GPDR) predictions via sampling...')
    mu_updated = []
    sigma_updated = []
    for (i, (lon, lat)) in enumerate(zip(lons, lats)):
        x_star_original = np.array([fixed_time, lon, lat])
        x_star = (x_star_original - x_mean) / x_std
        x_star = 2 / (1 + np.exp(-x_star)) - 1
        samples = sample_from_updated_model(x_star, n_samples=500, y_min=y_all.min().item(), y_max=y_all.max().item())
        mu_updated.append(np.mean(samples))
        sigma_updated.append(np.std(samples))
    mu_updated = np.array(mu_updated)
    sigma_updated = np.array(sigma_updated)
    mu_updated_original = mu_updated * y_std + y_mean
    sigma_updated_original = sigma_updated * y_std
    print(f'ĥ (GPDR) predictions: [{mu_updated_original.min():.1f}, {mu_updated_original.max():.1f}]°C, mean: {mu_updated_original.mean():.1f}°C')
    print(f'ĥ uncertainty: [{sigma_updated_original.min():.2f}, {sigma_updated_original.max():.2f}]°C (spatially varying)')
    print('\n=== CREATING SPATIAL VISUALIZATION ===')
    lon_range = lons.max() - lons.min()
    lat_range = lats.max() - lats.min()
    lon_grid = np.linspace(lons.min() - 0.02 * lon_range, lons.max() + 0.02 * lon_range, 100)
    lat_grid = np.linspace(lats.min() - 0.02 * lat_range, lats.max() + 0.02 * lat_range, 100)
    (lon_mesh, lat_mesh) = np.meshgrid(lon_grid, lat_grid)
    x_flat = lon_mesh.ravel()
    y_flat = lat_mesh.ravel()
    xy = np.column_stack([x_flat, y_flat])
    outside = np.ones(xy.shape[0], dtype=bool)
    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            inside_poly = points_in_polygon_path(xy, geom)
            outside &= ~inside_poly
        elif isinstance(geom, MultiPolygon):
            inside_any = np.zeros(xy.shape[0], dtype=bool)
            for part in geom.geoms:
                inside_any |= points_in_polygon_path(xy, part)
            outside &= ~inside_any
        else:
            continue
    mask = outside.reshape(lon_mesh.shape)
    X_grid_original = np.column_stack([np.full(lon_mesh.size, fixed_time), lon_mesh.ravel(), lat_mesh.ravel()])
    X_grid = (X_grid_original - x_mean) / x_std
    X_grid = 2 / (1 + np.exp(-X_grid)) - 1
    mu_g_tilde_grid = gam.predict(X_grid).reshape(lon_mesh.shape)
    sigma_g_tilde_grid = np.full_like(mu_g_tilde_grid, ORIGINAL_SIGMA_NORMAL)
    mu_g_tilde_grid_original = mu_g_tilde_grid * y_std + y_mean
    sigma_g_tilde_grid_original = sigma_g_tilde_grid * y_std
    mu_updated_grid = np.full(lon_mesh.shape, np.nan)
    sigma_updated_grid = np.full(lon_mesh.shape, np.nan)
    valid_idx = np.where(~mask.ravel())[0]
    n_samples_grid = 100
    for (k, idx) in enumerate(valid_idx):
        samples = sample_from_updated_model(X_grid[idx], n_samples=n_samples_grid, y_min=y_all.min().item(), y_max=y_all.max().item())
        mu_updated_grid.ravel()[idx] = np.mean(samples)
        sigma_updated_grid.ravel()[idx] = np.std(samples)
    mu_updated_grid_original = mu_updated_grid * y_std + y_mean
    sigma_updated_grid_original = sigma_updated_grid * y_std
    mu_g_tilde_masked = np.ma.array(mu_g_tilde_grid_original, mask=mask)
    mu_updated_masked = np.ma.array(mu_updated_grid_original, mask=mask)
    sigma_g_tilde_masked = np.ma.array(sigma_g_tilde_grid_original, mask=mask)
    sigma_updated_masked = np.ma.array(sigma_updated_grid_original, mask=mask)
    (fig, axes) = plt.subplots(2, 2, figsize=(14, 12))
    temp_p5 = np.percentile(temps_obs, 5)
    temp_p95 = np.percentile(temps_obs, 95)
    temp_median = np.median(temps_obs)
    from matplotlib.colors import TwoSlopeNorm
    temp_norm = TwoSlopeNorm(vmin=temps_obs.min(), vcenter=temp_median, vmax=temps_obs.max())
    print(f'Temperature color mapping: min={temps_obs.min():.1f}°C, p5={temp_p5:.1f}°C, median={temp_median:.1f}°C, p95={temp_p95:.1f}°C, max={temps_obs.max():.1f}°C')
    sigma_vmin = np.nanmin(sigma_updated_masked)
    sigma_vmax = np.nanmax(sigma_updated_masked)
    print(f'GPDR uncertainty color range: [{sigma_vmin:.2f}, {sigma_vmax:.2f}]°C')
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(lons, lats, c=temps_obs, cmap='coolwarm', norm=temp_norm, s=20, edgecolors='k', linewidths=0.3)
    gdf.boundary.plot(ax=ax1, color='k', linewidth=1.0)
    ax1.set_title(f'Observed Temperatures', fontweight='bold')
    ax1.set_xlabel('Longitude', fontsize=16)
    ax1.set_ylabel('Latitude', fontsize=16)
    ax1.set_aspect('equal')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='3%', pad=0.05)
    cb1 = fig.colorbar(sc1, cax=cax1)
    cb1.set_label('Temperature (°C)', fontsize=16)
    norm_positions = np.linspace(0, 1, 7)
    temp_tick_values = [temp_norm.inverse(p) for p in norm_positions]
    cb1.set_ticks(temp_tick_values)
    cb1.set_ticklabels([f'{t:.1f}' for t in temp_tick_values])
    ax2 = axes[0, 1]
    im2 = ax2.imshow(mu_g_tilde_masked, extent=[lon_mesh.min(), lon_mesh.max(), lat_mesh.min(), lat_mesh.max()], cmap='coolwarm', norm=temp_norm, aspect='equal', origin='lower')
    gdf.boundary.plot(ax=ax2, color='k', linewidth=1.0)
    ax2.set_title(f'$g_0$ (Normal GAM) Predictions', fontweight='bold')
    ax2.set_xlabel('Longitude', fontsize=16)
    ax2.set_ylabel('Latitude', fontsize=16)
    ax2.set_aspect('equal')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='3%', pad=0.05)
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label('Temperature (°C)', fontsize=16)
    cb2.set_ticks(temp_tick_values)
    cb2.set_ticklabels([f'{t:.1f}' for t in temp_tick_values])
    ax3 = axes[1, 0]
    im3 = ax3.imshow(mu_updated_masked, extent=[lon_mesh.min(), lon_mesh.max(), lat_mesh.min(), lat_mesh.max()], cmap='coolwarm', norm=temp_norm, aspect='equal', origin='lower')
    gdf.boundary.plot(ax=ax3, color='k', linewidth=1.0)
    ax3.set_title('ĥ (GPDR) Predictions', fontweight='bold')
    ax3.set_xlabel('Longitude', fontsize=16)
    ax3.set_ylabel('Latitude', fontsize=16)
    ax3.set_aspect('equal')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='3%', pad=0.05)
    cb3 = fig.colorbar(im3, cax=cax3)
    cb3.set_label('Temperature (°C)', fontsize=16)
    cb3.set_ticks(temp_tick_values)
    cb3.set_ticklabels([f'{t:.1f}' for t in temp_tick_values])
    ax4 = axes[1, 1]
    im4 = ax4.imshow(sigma_updated_masked, extent=[lon_mesh.min(), lon_mesh.max(), lat_mesh.min(), lat_mesh.max()], cmap='Reds', vmin=sigma_vmin, vmax=sigma_vmax, aspect='equal', origin='lower')
    gdf.boundary.plot(ax=ax4, color='k', linewidth=1.0)
    ax4.set_title('ĥ (GPDR) Uncertainty\n(Spatially Varying)', fontweight='bold')
    ax4.set_xlabel('Longitude', fontsize=16)
    ax4.set_ylabel('Latitude', fontsize=16)
    ax4.set_aspect('equal')
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='3%', pad=0.05)
    cb4 = fig.colorbar(im4, cax=cax4)
    cb4.set_label('Std Dev (°C)', fontsize=16)
    plt.tight_layout()
    plt.savefig('spatial_analysis_fixed_time_only.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def spatial_metrics(state):
    """Port of GPDR notebook cell 23 (one-based)."""
    X_fixed_time = state['X_fixed_time']
    get_base_g_stats = state['get_base_g_stats']
    mu_g_tilde_original = state['mu_g_tilde_original']
    mu_updated_original = state['mu_updated_original']
    np = state['np']
    temps_obs = state['temps_obs']
    y_mean = state['y_mean']
    y_std = state['y_std']
    def metrics(y_true, y_pred):
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        return (mae, rmse)
    mu_g = []
    for i in range(len(X_fixed_time)):
        x_star = X_fixed_time[i]
        (mu, _, _) = get_base_g_stats(x_star)
        mu_g.append(mu)
    mu_g = np.array(mu_g)
    mu_g_original = mu_g * y_std + y_mean
    (g_mae, g_rmse) = metrics(temps_obs, mu_g_original)
    (g_tilde_mae, g_tilde_rmse) = metrics(temps_obs, mu_g_tilde_original)
    (h_mae, h_rmse) = metrics(temps_obs, mu_updated_original)
    print('\n=== Prediction Error Summary ===')
    print(f'g (t-dist)     -> MAE: {g_mae:.3f} °C | RMSE: {g_rmse:.3f} °C | Total Error: {np.sum(np.abs(mu_g_original - temps_obs)):.1f}')
    print(f'g̃ (Normal GAM) -> MAE: {g_tilde_mae:.3f} °C | RMSE: {g_tilde_rmse:.3f} °C | Total Error: {np.sum(np.abs(mu_g_tilde_original - temps_obs)):.1f}')
    print(f'ĥ (GPDR)       -> MAE: {h_mae:.3f} °C | RMSE: {h_rmse:.3f} °C | Total Error: {np.sum(np.abs(mu_updated_original - temps_obs)):.1f}')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def metrics(state):
    """Port of GPDR notebook cell 24 (one-based)."""
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    format_elapsed_time = state['format_elapsed_time']
    gam = state['gam']
    model = state['model']
    norm = state['norm']
    np = state['np']
    x_test = state['x_test']
    y_all = state['y_all']
    y_mean = state['y_mean']
    y_std = state['y_std']
    y_test = state['y_test']
    import time
    weather_metric_start = time.perf_counter()
    n_test = len(x_test)
    updated_predictions_orig = []
    updated_intervals_orig_lower = []
    updated_intervals_orig_upper = []
    base_g_tilde_predictions_orig = []
    base_g_tilde_intervals_orig_lower = []
    base_g_tilde_intervals_orig_upper = []
    y_obs_orig = []
    for i in range(n_test):
        x_star_std = x_test[i].cpu().numpy()
        y_star_std = float(y_test[i].cpu().numpy().item())
        y_star_orig = y_star_std * y_std + y_mean
        y_obs_orig.append(y_star_orig)
        (yg, gg, gg_tilde, hhat, _, _) = model.predict_density(x_star_std, y_min=y_all.min().item(), y_max=y_all.max().item(), M=1000)
        dy = yg[1] - yg[0]
        h_hat_norm = hhat / np.trapz(hhat, dx=dy)
        mean_y_std = np.trapz(yg * h_hat_norm, dx=dy)
        mean_y_orig = mean_y_std * y_std + y_mean
        updated_predictions_orig.append(mean_y_orig)
        cdf = np.cumsum(h_hat_norm) * dy
        cdf = np.clip(cdf, 0, 1)
        lower_q = 0.025
        upper_q = 0.975
        lower_quantile_std = np.interp(lower_q, cdf, yg)
        upper_quantile_std = np.interp(upper_q, cdf, yg)
        lower_quantile_orig = lower_quantile_std * y_std + y_mean
        upper_quantile_orig = upper_quantile_std * y_std + y_mean
        updated_intervals_orig_lower.append(lower_quantile_orig)
        updated_intervals_orig_upper.append(upper_quantile_orig)
        x_star_std_reshape = np.array(x_star_std).reshape(1, -1)
        mu_std = gam.predict(x_star_std_reshape)[0]
        mu_orig = mu_std * y_std + y_mean
        base_g_tilde_predictions_orig.append(mu_orig)
        lower_std = norm.ppf(0.025, loc=mu_std, scale=ORIGINAL_SIGMA_NORMAL)
        upper_std = norm.ppf(0.975, loc=mu_std, scale=ORIGINAL_SIGMA_NORMAL)
        lower_orig = lower_std * y_std + y_mean
        upper_orig = upper_std * y_std + y_mean
        base_g_tilde_intervals_orig_lower.append(lower_orig)
        base_g_tilde_intervals_orig_upper.append(upper_orig)
    y_obs_orig = np.array(y_obs_orig)
    updated_predictions_orig = np.array(updated_predictions_orig)
    base_g_tilde_predictions_orig = np.array(base_g_tilde_predictions_orig)
    updated_intervals_orig_lower = np.array(updated_intervals_orig_lower)
    updated_intervals_orig_upper = np.array(updated_intervals_orig_upper)
    base_g_tilde_intervals_orig_lower = np.array(base_g_tilde_intervals_orig_lower)
    base_g_tilde_intervals_orig_upper = np.array(base_g_tilde_intervals_orig_upper)
    rmse_g_tilde_orig = np.sqrt(np.mean((base_g_tilde_predictions_orig - y_obs_orig) ** 2))
    rmse_updated_orig = np.sqrt(np.mean((updated_predictions_orig - y_obs_orig) ** 2))
    print('\n' + '=' * 70)
    print('MODEL COMPARISON DIAGNOSTICS (Weather Data - ORIGINAL SCALE)')
    print('=' * 70)
    print(f'\nData Scale Information:')
    print(f'   Observed test range: [{y_obs_orig.min():.2f}, {y_obs_orig.max():.2f}] °C')
    print('\nRMSE (Root Mean Squared Error) in °C:')
    print(f'   $g_0$ (Normal GAM): {rmse_g_tilde_orig:.4f} °C')
    print(f'   ĥ (GPDR):       {rmse_updated_orig:.4f} °C')
    print()
    coverage_g_tilde = np.mean((y_obs_orig >= base_g_tilde_intervals_orig_lower) & (y_obs_orig <= base_g_tilde_intervals_orig_upper))
    coverage_updated = np.mean((y_obs_orig >= updated_intervals_orig_lower) & (y_obs_orig <= updated_intervals_orig_upper))
    avg_length_g_tilde = np.mean(base_g_tilde_intervals_orig_upper - base_g_tilde_intervals_orig_lower)
    avg_length_updated = np.mean(updated_intervals_orig_upper - updated_intervals_orig_lower)
    print(f'95% CI Coverage:')
    print(f'   $g_0$ (Normal GAM): {coverage_g_tilde:.4f}')
    print(f'   ĥ (GPDR):       {coverage_updated:.4f}')
    print(f'\n95% CI Avg Width (°C):')
    print(f'   $g_0$ (Normal GAM): {avg_length_g_tilde:.4f} °C')
    print(f'   ĥ (GPDR):       {avg_length_updated:.4f} °C')
    weather_metric_time_seconds = time.perf_counter() - weather_metric_start
    gpdr_prediction_time_seconds = state.get('weather_logscore_time_seconds', 0.0) + weather_metric_time_seconds
    gpdr_prediction_time = format_elapsed_time(gpdr_prediction_time_seconds)
    print(f'GPDR prediction/evaluation time: {gpdr_prediction_time}')
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state


def diagnostics(state):
    """Port of GPDR notebook cell 30 (one-based)."""
    ORIGINAL_SIGMA_NORMAL = state['ORIGINAL_SIGMA_NORMAL']
    gam = state['gam']
    model = state['model']
    norm = state['norm']
    np = state['np']
    plt = state['plt']
    torch = state['torch']
    x_test = state['x_test']
    y_all = state['y_all']
    y_test = state['y_test']
    @torch.no_grad()
    def quantile_residual_diagnostic_weather(model, x_obs, y_obs, y_min, y_max, M=1200, eps=1e-10, subset_size=300, seed=123):
        """Compute quantile residuals for GPDR and the normal base model g_0 on a reproducible subset."""
        gpdr_pit = []
        gpdr_mean = []
        base_pit = []
        base_mean = []
        x_np = x_obs.detach().cpu().numpy()
        y_np = y_obs.detach().cpu().numpy().reshape(-1)
        n_obs_total = len(y_np)
        if subset_size is not None and subset_size < n_obs_total:
            rng = np.random.default_rng(seed)
            subset_idx = np.sort(rng.choice(n_obs_total, size=subset_size, replace=False))
            x_np = x_np[subset_idx]
            y_np = y_np[subset_idx]
        else:
            subset_idx = np.arange(n_obs_total)
        for (x_i, y_i) in zip(x_np, y_np):
            (y_grid, _, _, h_hat, _, _) = model.predict_density(x_i, y_min=y_min, y_max=y_max, M=M)
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
            mu_base_i = float(gam.predict(np.array(x_i).reshape(1, -1))[0])
            base_mean.append(mu_base_i)
            base_pit.append(norm.cdf(y_i, loc=mu_base_i, scale=ORIGINAL_SIGMA_NORMAL))
        gpdr_pit = np.clip(np.asarray(gpdr_pit), eps, 1.0 - eps)
        base_pit = np.clip(np.asarray(base_pit), eps, 1.0 - eps)
        gpdr_mean = np.asarray(gpdr_mean)
        base_mean = np.asarray(base_mean)
        gpdr_resid = norm.ppf(gpdr_pit)
        base_resid = norm.ppf(base_pit)
        return {'gpdr': {'pit': gpdr_pit, 'quantile_residual': gpdr_resid, 'fitted_mean': gpdr_mean, 'summary': {'mean': float(np.mean(gpdr_resid)), 'std': float(np.std(gpdr_resid, ddof=1)), 'q05': float(np.quantile(gpdr_resid, 0.05)), 'median': float(np.quantile(gpdr_resid, 0.5)), 'q95': float(np.quantile(gpdr_resid, 0.95)), 'n_used': int(len(gpdr_resid))}}, 'base_g0': {'pit': base_pit, 'quantile_residual': base_resid, 'fitted_mean': base_mean, 'summary': {'mean': float(np.mean(base_resid)), 'std': float(np.std(base_resid, ddof=1)), 'q05': float(np.quantile(base_resid, 0.05)), 'median': float(np.quantile(base_resid, 0.5)), 'q95': float(np.quantile(base_resid, 0.95)), 'n_used': int(len(base_resid))}}, 'subset_idx': subset_idx}
    weather_qr_diag = quantile_residual_diagnostic_weather(model, x_test, y_test, y_min=y_all.min().item(), y_max=y_all.max().item(), M=1000, subset_size=4000, seed=123)
    print('Weather quantile residual diagnostics')
    print('=' * 70)
    for model_name in ['base_g0', 'gpdr']:
        payload = weather_qr_diag[model_name]
        print()
        print(model_name)
        for (key, value) in payload['summary'].items():
            if isinstance(value, float):
                print(f'  {key:>6s}: {value:.6f}')
            else:
                print(f'  {key:>6s}: {value}')
    print()
    print(f"subset_size_used: {len(weather_qr_diag['subset_idx'])}")
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    plot_specs = [('base_g0', axes[0], 'Base $g_0$ quantile residuals', '#457B9D'), ('gpdr', axes[1], 'GPDR quantile residuals', '#2A9D8F')]
    for (key, ax, title, color) in plot_specs:
        fitted_mean = weather_qr_diag[key]['fitted_mean']
        residual = weather_qr_diag[key]['quantile_residual']
        ax.scatter(fitted_mean, residual, s=10, alpha=0.35, color=color, edgecolors='none')
        ax.axhline(0.0, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('Fitted mean')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Quantile residual')
    plt.tight_layout()
    plt.savefig('weather_quantile_residuals_vs_fitted_mean.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    state.update({key: value for key, value in locals().items() if key != "state"})
    return state

STAGES = (setup, seed, prepare, split, covariate_histogram, fit_base, base_scale, pit_scatter, base_histogram, base_summary, map_setup, base_effects, helpers, train, density_plots, log_score, prediction_helpers, time_effects, spatial_effects, spatial_metrics, metrics, diagnostics,)
