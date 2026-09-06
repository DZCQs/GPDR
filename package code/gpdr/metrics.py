import math

import numpy as np
from scipy.special import gammaln
from scipy.stats import beta as beta_dist
from scipy.stats import norm


TOY_REFERENCE = {
    "rmse_updated": 0.1615,
    "rmse_normal": 0.1965,
    "coverage_updated_0.50": 0.4981,
    "coverage_normal_0.50": 0.5250,
    "width_updated_0.50": 0.2028,
    "width_normal_0.50": 0.2565,
    "coverage_updated_0.80": 0.7987,
    "coverage_normal_0.80": 0.8053,
    "width_updated_0.80": 0.3862,
    "width_normal_0.80": 0.4874,
    "coverage_updated_0.90": 0.9009,
    "coverage_normal_0.90": 0.8935,
    "width_updated_0.90": 0.5004,
    "width_normal_0.90": 0.6255,
    "coverage_updated_0.95": 0.9533,
    "coverage_normal_0.95": 0.9384,
    "width_updated_0.95": 0.6061,
    "width_normal_0.95": 0.7454,
}

GINI_REFERENCE = {
    "loglik_base": 243.1235,
    "loglik_updated": 303.7593,
    "loglik_base_per_obs": 1.157731,
    "loglik_updated_per_obs": 1.446473,
    "rmse_base_mc": 0.077825,
    "rmse_updated_mc": 0.068868,
    "coverage_base_mc": 0.9381,
    "coverage_updated_mc": 0.9333,
    "avg_length_base_mc": 0.285313,
    "avg_length_updated_mc": 0.243938,
}

WEATHER_REFERENCE = {
    "log_density_updated": -10920.2537,
    "log_density_g_tilde": -13071.9185,
    "rmse_g_tilde_orig": 4.2602,
    "rmse_updated_orig": 4.0355,
    "coverage_g_tilde": 0.9551,
    "coverage_updated": 0.9548,
    "avg_length_g_tilde": 16.8838,
    "avg_length_updated": 15.0435,
}


def density_mean_interval(y_grid, density, level=0.95, *, cdf_rule="trapezoid"):
    if not 0 < level < 1:
        raise ValueError("level must lie between 0 and 1")
    y_grid, density = np.asarray(y_grid), np.asarray(density)
    if y_grid.ndim != 1 or len(y_grid) < 2 or density.shape != y_grid.shape:
        raise ValueError("require matching one-dimensional grid/density arrays of length >= 2")
    if not np.isfinite(y_grid).all() or not np.all(np.diff(y_grid) > 0):
        raise ValueError("response grid must be finite and strictly increasing")
    if not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError("density must be finite and nonnegative")
    dy = y_grid[1] - y_grid[0]
    integration = dict(dx=dy) if cdf_rule == "rectangle" else dict(x=y_grid)
    mass = np.trapz(density, **integration)
    if mass <= 0:
        raise ValueError("density must have positive mass on the response grid")
    dens = density / mass
    mean = np.trapz(y_grid * dens, **integration)
    if cdf_rule == "trapezoid":
        from scipy.integrate import cumulative_trapezoid
        cdf = cumulative_trapezoid(dens, y_grid, initial=0.)
        cdf = cdf / cdf[-1]
    elif cdf_rule == "rectangle":
        # Explicit compatibility with the original notebook's uniform-grid CDF.
        cdf = np.clip(np.cumsum(dens) * dy, 0.0, 1.0)
    else:
        raise ValueError("cdf_rule must be 'trapezoid' or 'rectangle'")
    lower_q = (1.0 - level) / 2.0
    upper_q = 1.0 - lower_q
    return mean, np.interp(lower_q, cdf, y_grid), np.interp(upper_q, cdf, y_grid)


def evaluate_predictions(y_true, predictions, *, level=0.95):
    """Four held-out metrics from public GPDR DensityPrediction objects.

    Responses and grids must share the same response scale. No density floor,
    response rescaling, data splitting, or model fitting is performed here.
    Use the example's evaluation recipe to reproduce its MC or unit conventions.
    """
    y_true = np.asarray(y_true).reshape(-1)
    means, lows, highs, logs = [], [], [], []
    for i, prediction in enumerate(predictions):
        if i >= len(y_true):
            raise ValueError("more predictions than observations")
        mean, lo, hi = density_mean_interval(prediction.y, prediction.density, level)
        means.append(mean); lows.append(lo); highs.append(hi)
        with np.errstate(divide="ignore"):
            logs.append(np.log(np.interp(y_true[i], prediction.y, prediction.density, left=0., right=0.)))
    if not len(y_true) or len(means) != len(y_true):
        raise ValueError("require one prediction per observation, with n > 0")
    means, lows, highs = map(np.asarray, (means, lows, highs))
    return dict(log_score=float(np.sum(logs)), mean_log_score=float(np.mean(logs)),
                RMSE=float(np.sqrt(np.mean((means-y_true)**2))),
                coverage=float(np.mean((y_true >= lows) & (y_true <= highs))),
                average_width=float(np.mean(highs-lows)))


def toy_metrics(model, data, *, M=1000, y_min=-0.3, y_max=3.0, max_test=None):
    x_test = data["x_test"].detach().cpu().numpy()
    y_test = data["y_test"].detach().cpu().numpy()
    if max_test:
        x_test = x_test[:max_test]
        y_test = y_test[:max_test]
    ci_levels = [0.50, 0.80, 0.90, 0.95]
    updated_preds = []
    normal_preds = []
    updated_intervals = {level: [[], []] for level in ci_levels}
    normal_intervals = {level: [[], []] for level in ci_levels}
    slope = float(data["slope"])
    sig = float(data["sig_base"])
    for x_star in x_test:
        prediction = model.predict_density(float(x_star), y_min=y_min, y_max=y_max, M=M)
        yg, hhat = prediction.y, prediction.density
        for level in ci_levels:
            mean, lo, hi = density_mean_interval(yg, hhat, level, cdf_rule="rectangle")
            if level == ci_levels[0]:
                updated_preds.append(mean)
            updated_intervals[level][0].append(lo)
            updated_intervals[level][1].append(hi)
            lower_q = (1.0 - level) / 2.0
            mu = slope * float(x_star)
            normal_intervals[level][0].append(mu + sig * norm.ppf(lower_q))
            normal_intervals[level][1].append(mu + sig * norm.ppf(1.0 - lower_q))
        normal_preds.append(slope * float(x_star))
    out = {
        "rmse_updated": float(np.sqrt(np.mean((np.asarray(updated_preds) - y_test) ** 2))),
        "rmse_normal": float(np.sqrt(np.mean((np.asarray(normal_preds) - y_test) ** 2))),
    }
    for level in ci_levels:
        ulo, uhi = map(np.asarray, updated_intervals[level])
        nlo, nhi = map(np.asarray, normal_intervals[level])
        suffix = f"{level:.2f}"
        out[f"coverage_updated_{suffix}"] = float(np.mean((y_test >= ulo) & (y_test <= uhi)))
        out[f"coverage_normal_{suffix}"] = float(np.mean((y_test >= nlo) & (y_test <= nhi)))
        out[f"width_updated_{suffix}"] = float(np.mean(uhi - ulo))
        out[f"width_normal_{suffix}"] = float(np.mean(nhi - nlo))
    return out


def gini_loglik_metrics(model, data, *, M=1000):
    df_test = data["df_test"]
    X_test = data["X_test"]
    X_scaled = data["X_gp_test_scaled"]
    beta_hat = data["beta_hat"]
    phi_hat = data["phi_hat"]
    loglik_base = 0.0
    loglik_updated = 0.0
    for i in range(len(df_test)):
        y_star = float(df_test["y_adj"].iloc[i])
        xrow = X_test.iloc[i].to_numpy(dtype=float)
        mu_star = float(df_test["mu_hat"].iloc[i])
        a = mu_star * phi_hat
        b = (1.0 - mu_star) * phi_hat
        loglik_base += gammaln(a + b) - gammaln(a) - gammaln(b) + (a - 1.0) * np.log(y_star) + (b - 1.0) * np.log(1.0 - y_star)
        from .paper.configuration import predict_gini_row
        prediction = predict_gini_row(model, data, X_row=X_test.iloc[i], df_row=df_test.iloc[i], M=M)
        yg, hhat = prediction.y, prediction.density
        loglik_updated += np.log(max(np.interp(y_star, yg, hhat), 1e-12))
    n = len(df_test)
    return {
        "loglik_base": float(loglik_base),
        "loglik_updated": float(loglik_updated),
        "loglik_base_per_obs": float(loglik_base / n),
        "loglik_updated_per_obs": float(loglik_updated / n),
    }


def gini_mc_metrics(model, data, *, n_samples=3000, M=1000, seed=None):
    # The notebook interleaves beta draws and grid draws using NumPy's global RNG.
    rng = np.random if seed is None else np.random.RandomState(seed)
    df_test = data["df_test"]
    X_test = data["X_test"]
    X_scaled = data["X_gp_test_scaled"]
    beta_hat = data["beta_hat"]
    phi_hat = data["phi_hat"]
    base_samples_all = []
    updated_samples_all = []
    for i in range(len(df_test)):
        xrow = X_test.iloc[i].to_numpy(dtype=float)
        from scipy.special import expit
        mu_star = expit(beta_hat @ xrow)
        a = mu_star * phi_hat
        b = (1.0 - mu_star) * phi_hat
        base_samples_all.append(beta_dist.rvs(a, b, size=n_samples, random_state=None if seed is None else rng))
        from .paper.configuration import predict_gini_row
        prediction = predict_gini_row(model, data, X_row=X_test.iloc[i], df_row=df_test.iloc[i], M=M)
        yg, hhat = prediction.y, prediction.density
        dy = np.diff(yg)
        wts = np.zeros_like(yg)
        wts[0] = dy[0] / 2.0
        wts[-1] = dy[-1] / 2.0
        wts[1:-1] = (dy[:-1] + dy[1:]) / 2.0
        prob = hhat * wts
        prob = prob / prob.sum()
        updated_samples_all.append(rng.choice(yg, size=n_samples, replace=True, p=prob))
    base = np.asarray(base_samples_all)
    updated = np.asarray(updated_samples_all)
    y = df_test["y_adj"].to_numpy()
    lower_base, upper_base = np.percentile(base, [2.5, 97.5], axis=1)
    lower_updated, upper_updated = np.percentile(updated, [2.5, 97.5], axis=1)
    return {
        "rmse_base_mc": float(np.sqrt(np.mean((y - base.mean(axis=1)) ** 2))),
        "rmse_updated_mc": float(np.sqrt(np.mean((y - updated.mean(axis=1)) ** 2))),
        "coverage_base_mc": float(np.mean((y >= lower_base) & (y <= upper_base))),
        "coverage_updated_mc": float(np.mean((y >= lower_updated) & (y <= upper_updated))),
        "avg_length_base_mc": float(np.mean(upper_base - lower_base)),
        "avg_length_updated_mc": float(np.mean(upper_updated - lower_updated)),
    }


def weather_metrics(model, data, *, M=1000, max_test=None):
    y_all = data["y"]
    x_test = data["x_test"].detach().cpu().numpy() if hasattr(data["x_test"], "detach") else np.asarray(data["x_test"])
    y_test = data["y_test"].detach().cpu().numpy() if hasattr(data["y_test"], "detach") else np.asarray(data["y_test"])
    if max_test:
        x_test = x_test[:max_test]
        y_test = y_test[:max_test]
    log_density_updated = 0.0
    log_density_g_tilde = 0.0
    updated_pred = []
    updated_lo = []
    updated_hi = []
    base_pred = []
    base_lo = []
    base_hi = []
    for x_star, y_star in zip(x_test, y_test):
        prediction = model.predict_density(x_star, y_min=float(y_all.min()), y_max=float(y_all.max()), M=M)
        if len(prediction) == 6:
            yg, gvals, gtilde, hhat, _, _ = prediction
        else:
            yg, gvals, hhat, _, _ = prediction
        h_interp = np.interp(float(y_star), yg, hhat)
        log_density_updated += np.log(h_interp)
        dy = yg[1] - yg[0]
        density = hhat / np.trapz(hhat, dx=dy)
        mean = np.trapz(yg * density, dx=dy)
        cdf = np.clip(np.cumsum(density) * dy, 0, 1)
        lo = np.interp(0.025, cdf, yg)
        hi = np.interp(0.975, cdf, yg)
        updated_pred.append(mean * data["y_std"] + data["y_mean"])
        updated_lo.append(lo * data["y_std"] + data["y_mean"])
        updated_hi.append(hi * data["y_std"] + data["y_mean"])
        mu = data["gam"].predict(np.asarray(x_star).reshape(1, -1))[0]
        if len(prediction) != 6:
            gtilde = (1.0 / (math.sqrt(2.0 * math.pi) * data["sigma_normal"])) * np.exp(-0.5 * ((yg - mu) / data["sigma_normal"]) ** 2)
        log_density_g_tilde += np.log(np.interp(float(y_star), yg, gtilde))
        base_pred.append(mu * data["y_std"] + data["y_mean"])
        base_lo.append(norm.ppf(0.025, loc=mu, scale=data["sigma_normal"]) * data["y_std"] + data["y_mean"])
        base_hi.append(norm.ppf(0.975, loc=mu, scale=data["sigma_normal"]) * data["y_std"] + data["y_mean"])
    y_obs = np.asarray([float(y_star) * data["y_std"] + data["y_mean"] for y_star in y_test])
    updated_pred = np.asarray(updated_pred)
    base_pred = np.asarray(base_pred)
    updated_lo = np.asarray(updated_lo)
    updated_hi = np.asarray(updated_hi)
    base_lo = np.asarray(base_lo)
    base_hi = np.asarray(base_hi)
    return {
        "log_density_updated": float(log_density_updated),
        "log_density_g_tilde": float(log_density_g_tilde),
        "rmse_g_tilde_orig": float(np.sqrt(np.mean((base_pred - y_obs) ** 2))),
        "rmse_updated_orig": float(np.sqrt(np.mean((updated_pred - y_obs) ** 2))),
        "coverage_g_tilde": float(np.mean((y_obs >= base_lo) & (y_obs <= base_hi))),
        "coverage_updated": float(np.mean((y_obs >= updated_lo) & (y_obs <= updated_hi))),
        "avg_length_g_tilde": float(np.mean(base_hi - base_lo)),
        "avg_length_updated": float(np.mean(updated_hi - updated_lo)),
    }


def compare_to_reference(metrics, reference):
    return {key: {"value": float(metrics[key]), "reference": float(reference[key]), "diff": float(metrics[key] - reference[key])} for key in reference if key in metrics}
