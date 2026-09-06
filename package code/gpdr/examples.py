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
from .models import GPDR, BetaGPMixtureDiag, LogisticGPMixtureDiag


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
    y = x + x**2 + torch.randn(n, generator=g) * (0.2 * x + 0.05)
    return x, y


def prepare_toy_data(n_train=5000, n_test=50000, seed_data=3, seed_split=2):
    from .paper.runner import new_state
    from .paper import toy
    _, data = new_state("toy", ".")
    data.update(n_train=n_train, n_test=n_test, seed_data=seed_data, seed_split=seed_split)
    toy.prepare(data)
    from .paper.configuration import toy_model_inputs
    data["base"] = toy_model_inputs(data)["base"]
    return data


def build_toy_model(data=None, C=2, m_induce=150, beta=1031.17, seed=2, kp=None):
    from .paper.configuration import toy_model_inputs
    data = data if data is not None else prepare_toy_data()
    return GPDR(**toy_model_inputs(data, C=C, m_induce=m_induce, beta=beta, kp=kp))


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
        imputation="mean",
        entropy_correction=False,
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
    from .paper.runner import new_state
    from .paper import gini
    _, data = new_state("gini", project_root)
    for stage in (gini.setup, gini.load_data, gini.split, gini.fit_base, gini.coefficients, gini.base_means, gini.prepare):
        stage(data)
    data["X_gp_test_scaled"] = ((data["X_gp_test"] - data["x_min"]) / (data["x_max"] - data["x_min"] + 1e-12)).to_numpy(dtype=float)
    return data


def build_gini_model(data, C=2, m_induce=200, beta=493.61, seed=2, kp=None):
    from .paper.configuration import gini_model_inputs
    return GPDR(**gini_model_inputs(data, C=C, m_induce=m_induce, beta=beta, seed=seed, kp=kp))


def load_gini_saved(project_root):
    root = _root(project_root) / "giniindex_thickertail"
    return ExampleBundle("gini", root, load_legacy_model(root / "optimized_model_gini.pkl"), None)


def load_weather_saved(project_root):
    root = _root(project_root) / "weatherdata_thickertail"
    return ExampleBundle("weather", root, load_legacy_model(root / "optimized_model_weather.pkl"), None)


def prepare_weather_data(project_root):
    from .paper.runner import new_state
    from .paper import weather
    _, data = new_state("weather", project_root)
    for stage in (weather.setup, weather.seed, weather.prepare, weather.split, weather.fit_base, weather.base_scale, weather.helpers):
        stage(data)
    data["sigma_normal"] = data["ORIGINAL_SIGMA_NORMAL"]
    data["scale_t"] = data["SCALE_T"]
    data["base"] = StudentTGamBase(data["gam"].predict, data["SCALE_T"], data["DF"])
    return data


def build_weather_model(data, C=2, m_induce=1000, beta=29.6, kp=None, device=None):
    from .paper.configuration import weather_model_inputs
    device = torch.device(device) if device is not None else torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cpu":
        torch.set_default_dtype(torch.float32)
    x = torch.as_tensor(data["x_train"], dtype=torch.float32, device=device)
    y = torch.as_tensor(data["y_train"], dtype=torch.float32, device=device)
    data["x_test"] = torch.as_tensor(data["x_test"], dtype=torch.float32, device=device)
    data["y_test"] = torch.as_tensor(data["y_test"], dtype=torch.float32, device=device)
    return GPDR(**weather_model_inputs(dict(data, x_train=x, y_train=y), C=C,
                                      m_induce=m_induce, beta=beta, kp=kp)).to(device)
