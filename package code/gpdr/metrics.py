import math

import numpy as np
from scipy.special import gammaln
from scipy.stats import beta as beta_dist
from scipy.stats import norm


TOY_REFERENCE = {
    "rmse_updated": 0.1161,
    "rmse_normal": 0.1613,
    "coverage_updated_0.50": 0.5289,
    "coverage_normal_0.50": 0.4928,
    "width_updated_0.50": 0.1397,
    "width_normal_0.50": 0.2084,
    "coverage_updated_0.80": 0.8116,
    "coverage_normal_0.80": 0.8108,
    "width_updated_0.80": 0.2656,
    "width_normal_0.80": 0.3960,
    "coverage_updated_0.90": 0.9066,
    "coverage_normal_0.90": 0.9007,
    "width_updated_0.90": 0.3447,
    "width_normal_0.90": 0.5083,
    "coverage_updated_0.95": 0.9569,
    "coverage_normal_0.95": 0.9418,
    "width_updated_0.95": 0.4204,
    "width_normal_0.95": 0.6057,
}

GINI_REFERENCE = {
    "loglik_base": 243.1235,
    "loglik_updated": 285.3993,
    "loglik_base_per_obs": 1.157731,
    "loglik_updated_per_obs": 1.359044,
    "rmse_base_mc": 0.077712,
    "rmse_updated_mc": 0.070693,
    "coverage_base_mc": 0.9286,
    "coverage_updated_mc": 0.9571,
    "avg_length_base_mc": 0.285190,
    "avg_length_updated_mc": 0.277214,
}

WEATHER_REFERENCE = {
    "log_density_updated": -11100.2877,
    "log_density_g_tilde": -13071.9185,
    "rmse_g_tilde_orig": 4.2602,
    "rmse_updated_orig": 4.0282,
    "coverage_g_tilde": 0.9551,
    "coverage_updated": 0.9607,
    "avg_length_g_tilde": 16.8838,
    "avg_length_updated": 15.3801,
}


def density_mean_interval(y_grid, density, level=0.95):
    dy = y_grid[1] - y_grid[0]
    dens = density / np.trapz(density, dx=dy)
    mean = np.trapz(y_grid * dens, dx=dy)
    cdf = np.clip(np.cumsum(dens) * dy, 0.0, 1.0)
    lower_q = (1.0 - level) / 2.0
    upper_q = 1.0 - lower_q
    return mean, np.interp(lower_q, cdf, y_grid), np.interp(upper_q, cdf, y_grid)


def toy_metrics(model, data, *, M=1000, y_min=-0.2, y_max=2.5, max_test=None):
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
        yg, _, hhat, _, _ = model.predict_density(float(x_star), y_min=y_min, y_max=y_max, M=M)
        for level in ci_levels:
            mean, lo, hi = density_mean_interval(yg, hhat, level)
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
        mu_star = 1.0 / (1.0 + np.exp(-(xrow @ beta_hat)))
        a = mu_star * phi_hat
        b = (1.0 - mu_star) * phi_hat
        loglik_base += gammaln(a + b) - gammaln(a) - gammaln(b) + (a - 1.0) * np.log(y_star) + (b - 1.0) * np.log1p(-y_star)
        yg, _, hhat, _, _ = model.predict_density_scaled(X_scaled[i], mu_star, phi_hat, y_min=1e-3, y_max=1 - 1e-3, M=M)
        loglik_updated += np.log(max(np.interp(y_star, yg, hhat), 1e-12))
    n = len(df_test)
    return {
        "loglik_base": float(loglik_base),
        "loglik_updated": float(loglik_updated),
        "loglik_base_per_obs": float(loglik_base / n),
        "loglik_updated_per_obs": float(loglik_updated / n),
    }


def gini_mc_metrics(model, data, *, n_samples=2000, M=1000, seed=None):
    rng = np.random.default_rng(seed)
    df_test = data["df_test"]
    X_test = data["X_test"]
    X_scaled = data["X_gp_test_scaled"]
    beta_hat = data["beta_hat"]
    phi_hat = data["phi_hat"]
    base_samples_all = []
    updated_samples_all = []
    for i in range(len(df_test)):
        xrow = X_test.iloc[i].to_numpy(dtype=float)
        mu_star = 1.0 / (1.0 + np.exp(-(xrow @ beta_hat)))
        a = mu_star * phi_hat
        b = (1.0 - mu_star) * phi_hat
        base_samples_all.append(beta_dist.rvs(a, b, size=n_samples, random_state=rng))
        yg, _, hhat, _, _ = model.predict_density_scaled(X_scaled[i], mu_star, phi_hat, y_min=1e-3, y_max=1 - 1e-3, M=M)
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
    x_test = np.asarray(data["x_test"])
    y_test = np.asarray(data["y_test"])
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
        yg, gvals, hhat, _, _ = model.predict_density(x_star, y_min=float(y_all.min()), y_max=float(y_all.max()), M=M)
        h_interp = max(np.interp(float(y_star), yg, hhat), 1e-12)
        log_density_updated += np.log(h_interp)
        mean, lo, hi = density_mean_interval(yg, hhat, 0.95)
        updated_pred.append(mean * data["y_std"] + data["y_mean"])
        updated_lo.append(lo * data["y_std"] + data["y_mean"])
        updated_hi.append(hi * data["y_std"] + data["y_mean"])
        mu = float(data["gam"].predict(np.asarray(x_star).reshape(1, -1))[0])
        gtilde = (1.0 / (math.sqrt(2.0 * math.pi) * data["sigma_normal"])) * np.exp(-0.5 * ((yg - mu) / data["sigma_normal"]) ** 2)
        log_density_g_tilde += np.log(max(np.interp(float(y_star), yg, gtilde), 1e-12))
        base_pred.append(mu * data["y_std"] + data["y_mean"])
        base_lo.append(norm.ppf(0.025, loc=mu, scale=data["sigma_normal"]) * data["y_std"] + data["y_mean"])
        base_hi.append(norm.ppf(0.975, loc=mu, scale=data["sigma_normal"]) * data["y_std"] + data["y_mean"])
    y_obs = y_test * data["y_std"] + data["y_mean"]
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

