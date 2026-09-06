import numpy as np
from scipy.stats import norm
import torch

from gpdr.metrics import weather_metrics
from gpdr.paper import weather


class FixedGam:
    def predict(self, x):
        return 0.1 * np.asarray(x)[:, 0]


class FixedDensity:
    def predict_density(self, x, y_min, y_max, M):
        y = np.linspace(y_min, y_max, M).astype(np.float32)
        mu = FixedGam().predict(np.asarray(x).reshape(1, -1))[0]
        base = norm.pdf(y, mu, 0.8).astype(np.float32)
        updated = norm.pdf(y, mu, 0.25).astype(np.float32)
        return y, base, base, updated, np.ones_like(y), norm.cdf(y)


def test_weather_metric_helper_matches_native_evaluation():
    data = dict(model=FixedDensity(), gam=FixedGam(), np=np, norm=norm,
                x_test=torch.zeros(5, 3, dtype=torch.float32),
                y_test=torch.tensor([-2., -1., .2, .8, 2.], dtype=torch.float32),
                y_all=np.array([-3., 3.]), y=np.array([-3., 3.]),
                y_std=np.float64(7.319), y_mean=np.float64(11.71),
                ORIGINAL_SIGMA_NORMAL=0.8, sigma_normal=0.8)
    weather.log_score(data)
    weather.metrics(data)
    actual = weather_metrics(data['model'], data)
    mapping = {'log_density_updated': 'estimated_log_density_vals',
               'log_density_g_tilde': 'base_tilde_log_density_vals'}
    for key, value in actual.items():
        assert value == data[mapping.get(key, key)], key
