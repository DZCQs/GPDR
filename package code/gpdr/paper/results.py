"""Metric tables assembled from the actual notebook-equivalent evaluations."""
import numpy as np
import pandas as pd


def metric_table(example, state):
    if example == "toy":
        n = len(state["x_test"])
        scores = (state["base_normal_log_density_vals"], state["estimated_log_density_vals"])
        errors = (state["rmse_normal"], state["rmse_updated"])
        coverage = (state["coverage_normal"], state["coverage_updated"])
        widths = (state["length_normal"], state["length_updated"])
    elif example == "gini":
        n = len(state["df_test"])
        scores = (state["loglik_base"], state["loglik_updated"])
        errors = (state["rmse_base"], state["rmse_updated"])
        coverage = (state["coverage_base"], state["coverage_updated"])
        widths = (state["avg_length_base"], state["avg_length_updated"])
    elif example == "weather":
        n = len(state["x_test"])
        scores = (state["base_tilde_log_density_vals"], state["estimated_log_density_vals"])
        errors = (state["rmse_g_tilde_orig"], state["rmse_updated_orig"])
        coverage = (state["coverage_g_tilde"], state["coverage_updated"])
        widths = (state["avg_length_g_tilde"], state["avg_length_updated"])
    else:
        raise ValueError(f"Unknown example: {example}")
    return pd.DataFrame({"method": ["parametric model", "GPDR"], "log_score": scores,
                         "mean_log_score": np.asarray(scores)/n, "RMSE": errors,
                         "coverage_95": coverage, "average_width_95": widths})
