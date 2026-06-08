from pathlib import Path

import torch

from gpdr.examples import load_gini_saved, load_toy_saved, load_weather_saved


PROJECT_ROOT = Path("/Users/zichuanchen/Desktop/gp_density_regression")


def _assert_objective(bundle):
    F, info = bundle.model.forward_objective()
    assert torch.isfinite(F)
    assert "KL" in info
    assert torch.isfinite(info["KL"])


def test_load_toy_saved_model():
    _assert_objective(load_toy_saved(PROJECT_ROOT))


def test_load_gini_saved_model():
    _assert_objective(load_gini_saved(PROJECT_ROOT))


def test_load_weather_saved_model():
    _assert_objective(load_weather_saved(PROJECT_ROOT))

