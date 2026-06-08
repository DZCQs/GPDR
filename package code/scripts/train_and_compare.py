#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from gpdr.examples import (
    build_gini_model,
    build_toy_model,
    build_weather_model,
    prepare_gini_data,
    prepare_toy_data,
    prepare_weather_data,
)
from gpdr.metrics import (
    GINI_REFERENCE,
    TOY_REFERENCE,
    WEATHER_REFERENCE,
    compare_to_reference,
    gini_loglik_metrics,
    gini_mc_metrics,
    toy_metrics,
    weather_metrics,
)
from gpdr.training import fit_adam


def print_comparison(title, metrics, reference):
    print("\n" + title)
    print("=" * len(title))
    for key, row in compare_to_reference(metrics, reference).items():
        print(f"{key:28s} value={row['value']: .8f} reference={row['reference']: .8f} diff={row['diff']: .8f}")


def run_toy(args):
    torch.set_default_dtype(torch.float64)
    data = prepare_toy_data()
    torch.manual_seed(2)
    np.random.seed(2)
    model = build_toy_model(data)
    model, trace = fit_adam(model, steps=args.steps or 400000, lr=0.5, average_last=5000, verbose_every=args.verbose_every)
    metrics = toy_metrics(model, data, M=args.M, max_test=args.max_test)
    print(f"final_objective={trace['F'][-1]:.12g}")
    print_comparison("TOY METRICS", metrics, TOY_REFERENCE)
    return metrics


def run_gini(args):
    torch.set_default_dtype(torch.float64)
    data = prepare_gini_data(args.project_root)
    torch.manual_seed(0)
    np.random.seed(0)
    model = build_gini_model(data)
    model, trace = fit_adam(model, steps=args.steps or 2000, lr=0.5, average_last=0, verbose_every=args.verbose_every)
    metrics = gini_loglik_metrics(model, data, M=args.M)
    metrics.update(gini_mc_metrics(model, data, n_samples=args.n_samples, M=args.M, seed=args.mc_seed))
    print(f"final_objective={trace['F'][-1]:.12g}")
    print_comparison("GINI METRICS", metrics, GINI_REFERENCE)
    return metrics


def run_weather(args):
    data = prepare_weather_data(args.project_root)
    device = torch.device(args.device)
    if device.type != "cpu":
        torch.set_default_dtype(torch.float32)
    torch.manual_seed(1)
    np.random.seed(1)
    model = build_weather_model(data).to(device)
    if hasattr(model, "logits"):
        with torch.no_grad():
            model.logits.data.fill_(0.0)
        model.logits.requires_grad_(False)
    model, trace = fit_adam(model, steps=args.steps or 5000, lr=0.5, average_last=500, verbose_every=args.verbose_every)
    model = model.cpu()
    metrics = weather_metrics(model, data, M=args.M, max_test=args.max_test)
    print(f"final_objective={trace['F'][-1]:.12g}")
    print_comparison("WEATHER METRICS", metrics, WEATHER_REFERENCE)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Retrain GPDR examples and compare notebook metrics.")
    parser.add_argument("--project-root", default="/Users/zichuanchen/Desktop/gp_density_regression")
    parser.add_argument("--example", choices=["toy", "gini", "weather"], required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--M", type=int, default=1000)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--mc-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose-every", type=int, default=5000)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    if args.example == "toy":
        metrics = run_toy(args)
    elif args.example == "gini":
        metrics = run_gini(args)
    else:
        metrics = run_weather(args)

    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
