from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit

from .base import StudentTGamBase, StudentTLinearBase, StudentTLinearIndexBase
from .compat import load_legacy_model
from .kernels import KernelParams
from .models import BetaGPMixtureDiag, LogisticGPMixtureDiag


@dataclass
class ExampleBundle:
    name: str
    root: Path
    model: object | None = None
    data: dict | None = None


def _root(project_root):
    return Path(project_root).expanduser().resolve()


def make_toy(n=700, seed=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g)
    y = x + x**2 + torch.randn(n, generator=g) * (0.2 * x)
    return x, y


def prepare_toy_data(n_train=5000, n_test=50000, seed_data=3, seed_split=2):
    torch.manual_seed(2)
    np.random.seed(2)
    x_all, y_all = make_toy(n_train + n_test, seed_data)
    rand_indices = torch.randperm(x_all.size(0), generator=torch.Generator().manual_seed(seed_split))
    x, y = x_all[rand_indices[:n_train]], y_all[rand_indices[:n_train]]
    x_test, y_test = x_all[rand_indices[n_train:]], y_all[rand_indices[n_train:]]
    slope = np.linalg.lstsq(x.numpy().reshape(-1, 1), y.numpy(), rcond=None)[0][0]
    residuals = y.numpy() - slope * x.numpy()
    sig_base = np.std(residuals, ddof=1)
    base = StudentTLinearBase(float(slope), float(sig_base * 3.0), 3.0)
    return {"x": x, "y": y, "x_test": x_test, "y_test": y_test, "slope": slope, "sig_base": sig_base, "base": base}


def build_toy_model(data=None, C=1, m_induce=150, beta=680.09, seed=2):
    data = data or prepare_toy_data()
    x = data["x"].to(torch.get_default_dtype())
    y = data["y"].to(torch.get_default_dtype())
    return LogisticGPMixtureDiag(x, y, base=data["base"], C=C, m_induce=m_induce, beta=beta, seed=seed, inducing="grid")


def make_multix_toy(n=700, d=2, seed=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=g)
    y = x[:, 0] + x[:, 0] ** 2 + torch.randn(n, generator=g) * (0.2 * x[:, 0])
    return x, y


def multix_base_index(x):
    if torch.is_tensor(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x[:, 0] + 0.1 * x[:, 1]
    x = np.asarray(x)
    return x[..., 0] + 0.1 * x[..., 1]


def prepare_multix_toy_data(n_total=50000, seed_data=3, seed_split=4, device=None):
    torch.manual_seed(2)
    np.random.seed(2)
    x_all, y_all = make_multix_toy(n=n_total, d=2, seed=seed_data)
    rand_indices = torch.randperm(x_all.size(0), generator=torch.Generator().manual_seed(seed_split))
    n_test = x_all.size(0) // 10
    x = x_all[rand_indices[:-n_test]]
    y = y_all[rand_indices[:-n_test]]
    x_test = x_all[rand_indices[-n_test:]]
    y_test = y_all[rand_indices[-n_test:]]
    if device is not None:
        x = x.to(device)
        y = y.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

    s_train = multix_base_index(x).cpu().numpy().reshape(-1, 1)
    y_train_np = y.cpu().numpy()
    b_base = float(np.linalg.lstsq(s_train, y_train_np, rcond=None)[0][0])
    residuals = y_train_np - b_base * s_train.ravel()
    sigma_normal = float(np.std(residuals, ddof=1))
    base = StudentTLinearIndexBase(
        slope=b_base,
        coefficients=np.array([1.0, 0.1]),
        scale=3.0 * sigma_normal,
        df=3.0,
    )
    return {
        "x": x,
        "y": y,
        "x_test": x_test,
        "y_test": y_test,
        "b_base": b_base,
        "sigma_normal": sigma_normal,
        "scale_t": 3.0 * sigma_normal,
        "df_t": 3.0,
        "base": base,
    }


def build_multix_toy_model(data=None, C=1, m_induce=15**2, beta=2200.0, seed=2):
    data = data or prepare_multix_toy_data()
    x = data["x"].to(torch.get_default_dtype())
    y = data["y"].to(torch.get_default_dtype())
    kp = KernelParams(
        log_sigma2=float(np.log(0.25**2)),
        log_lx2=torch.log(torch.full((x.shape[1],), 0.28**2, dtype=x.dtype, device=x.device)),
        log_lz2=float(np.log(0.45**2)),
    )
    return LogisticGPMixtureDiag(
        x,
        y,
        base=data["base"],
        C=C,
        m_induce=m_induce,
        beta=beta,
        seed=seed,
        kp=kp,
        inducing="unit_grid",
        scale_hyvarinen_by_n=False,
    )


def load_toy_saved(project_root):
    root = _root(project_root) / "toyexample_thickertail"
    return ExampleBundle("toy", root, load_legacy_model(root / "optimized_model.pkl"), prepare_toy_data())


def _gini_neg_loglik(params, y, X):
    from scipy.special import gammaln

    beta = params[:-1]
    phi = np.exp(params[-1])
    mu = expit(X @ beta)
    eps = 1e-9
    y = np.clip(y, eps, 1.0 - eps)
    a = mu * phi
    b = (1.0 - mu) * phi
    ll = (
        gammaln(phi)
        - gammaln(a)
        - gammaln(b)
        + (a - 1.0) * np.log(y)
        + (b - 1.0) * np.log1p(-y)
    )
    return -float(np.sum(ll))


def prepare_gini_data(project_root):
    root = _root(project_root) / "giniindex_thickertail"
    df = pd.read_csv(root / "reg_df.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    cont_cols = ["log_gdppc", "urban", "unemp", "trade", "year"]
    X_cont = df[cont_cols].copy()
    X = X_cont.copy()
    X.insert(0, "Intercept", 1.0)
    y = df["y_adj"].to_numpy(dtype=float)
    X_mat = X.to_numpy(dtype=float)
    rng = np.random.default_rng(123)
    perm = rng.permutation(X_mat.shape[0])
    n_train = int(0.9 * X_mat.shape[0])
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    X_mat_train = X_mat[train_idx]
    X_mat_test = X_mat[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    init = np.zeros(X_mat_train.shape[1] + 1)
    res = minimize(_gini_neg_loglik, init, args=(y_train, X_mat_train), method="BFGS")
    beta_hat = res.x[:-1]
    phi_hat = float(np.exp(res.x[-1]))
    mu_hat_train = expit(X_mat_train @ beta_hat)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    X_gp_train = df_train[cont_cols].copy()
    X_gp_test = df_test[cont_cols].copy()
    X_gp_all = df[cont_cols].copy()
    x_min = X_gp_all.min(axis=0)
    x_max = X_gp_all.max(axis=0)
    X_gp_scaled = ((X_gp_train - x_min) / (x_max - x_min + 1e-12)).to_numpy(dtype=float)
    X_gp_test_scaled = ((X_gp_test - x_min) / (x_max - x_min + 1e-12)).to_numpy(dtype=float)
    mu_hat_test = expit(X_mat_test @ beta_hat)
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["mu_hat"] = mu_hat_train
    df_test["mu_hat"] = mu_hat_test
    return {
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "X_gp_train": X_gp_train,
        "X_gp_test": X_gp_test,
        "y": y,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_gp_scaled": X_gp_scaled,
        "X_gp_test_scaled": X_gp_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "beta_hat": beta_hat,
        "phi_hat": phi_hat,
        "mu_hat_train": mu_hat_train,
        "phi_base": phi_hat * 2.0,
    }


def build_gini_model(data, C=3, m_induce=400, beta=2.66, seed=2):
    x = torch.from_numpy(data["X_gp_scaled"]).to(torch.float64)
    y = torch.from_numpy(np.clip(data["y_train"], 1e-6, 1.0 - 1e-6)).to(torch.float64)
    kp = KernelParams(
        log_sigma2=float(np.log(0.9168**2)),
        log_lx2=torch.log(torch.full((x.shape[1],), 0.2117**2, dtype=torch.float64)),
        log_lz2=float(np.log(0.7829**2)),
    )
    return BetaGPMixtureDiag(x, y, mu_base_train=data["mu_hat_train"], phi_base=data["phi_base"], C=C, m_induce=m_induce, beta=beta, seed=seed, kp=kp)


def load_gini_saved(project_root):
    root = _root(project_root) / "giniindex_thickertail"
    return ExampleBundle("gini", root, load_legacy_model(root / "optimized_model_gini.pkl"), None)


def load_weather_saved(project_root):
    root = _root(project_root) / "weatherdata_thickertail"
    return ExampleBundle("weather", root, load_legacy_model(root / "optimized_model_weather.pkl"), None)


def prepare_weather_data(project_root):
    from pygam import LinearGAM, intercept, s, te

    root = _root(project_root) / "weatherdata_thickertail"
    torch.manual_seed(1)
    np.random.seed(1)
    df = pd.read_csv(root / "weather_data.csv", index_col="date")
    df.index = pd.to_datetime(df.index)
    df["t"] = (df.index - df.index[0]) / pd.Timedelta(days=1)
    df["t"] = df["t"] / df["t"].max()
    df = df[(df["t"] > 1 / 6) & (df["t"] < 2 / 6)].copy()
    df["t"] = (df["t"] - df["t"].min()) / (df["t"].max() - df["t"].min())
    X_original = df[["t", "longitude", "latitude"]].values
    y_original = df["temperature"].values
    x_mean = X_original.mean(axis=0)
    x_std = X_original.std(axis=0)
    X = (X_original - x_mean) / x_std
    X = (2.0 / (1.0 + np.exp(-X))) - 1.0
    y_mean = y_original.mean()
    y_std = y_original.std()
    y = (y_original - y_mean) / y_std
    rand_indices = torch.randperm(X.shape[0], generator=torch.Generator())
    split = X.shape[0] // 10
    x_train, y_train = X[rand_indices[:-split]], y[rand_indices[:-split]]
    x_test, y_test = X[rand_indices[-split:]], y[rand_indices[-split:]]
    gam = LinearGAM(intercept + s(0, n_splines=5) + te(1, 2, n_splines=[5, 5]))
    gam.fit(x_train, y_train)
    mu = gam.predict(X)
    sigma_normal = float(np.std(y - mu))
    scale_t = sigma_normal * 3.0
    base = StudentTGamBase(gam.predict, scale_t, 3.0)
    return {
        "df": df,
        "X": X,
        "y": y,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "gam": gam,
        "sigma_normal": sigma_normal,
        "scale_t": scale_t,
        "base": base,
    }


def build_weather_model(data, C=1, m_induce=1000, beta=4.8):
    x = torch.as_tensor(data["x_train"], dtype=torch.float32)
    y = torch.as_tensor(data["y_train"], dtype=torch.float32)
    kp = KernelParams(
        log_sigma2=float(np.log(0.4240**2)),
        log_lx2=torch.log(torch.full((x.shape[1],), 0.3290**2, dtype=torch.float32)),
        log_lz2=float(np.log(0.5469**2)),
    )
    return LogisticGPMixtureDiag(
        x,
        y,
        base=data["base"],
        C=C,
        m_induce=m_induce,
        beta=beta,
        kp=kp,
        inducing="kmeans",
        scale_hyvarinen_by_n=True,
    )
