"""Reproduce the toy figures and metrics using the public model/trainer.

Run from gpdr_package: python examples/toy_public_api.py
No run_example call, notebook execution, stored fitted model, or reference metric
is used. Example-specific code below supplies data, configuration, and plots.
"""
import math
from pathlib import Path

import numpy as np
import torch

from gpdr import GPDR, KernelParams, fit_adam, prepare_toy_data, tensor_grid
from gpdr.paper import toy
from gpdr.paper.results import metric_table
from gpdr.paper.runner import output_directory


def reproduce(output_dir):
    data = prepare_toy_data()
    x, y = data['x'], data['y']
    torch.manual_seed(2)
    np.random.seed(2)
    inducing_points = tensor_grid(
        torch.linspace(x.min(), x.max(), 12),
        torch.linspace(-0.05, 1.05, 12),
    )
    model = GPDR(
        x, y,
        base=data['base'],
        inducing_points=inducing_points,
        C=2,
        beta=1031.17,
        kp=KernelParams(
            log_sigma2=math.log(0.9199**2),
            log_lx2=math.log(0.0786**2),
            log_lz2=math.log(0.3363**2),
        ),
    )
    model, trace = fit_adam(model, steps=300000, lr=0.5, average_last=5000,
                           averaging='running', trace_at='before', verbose_every=10000)
    data.update(model=model, trace=trace)
    with output_directory(output_dir):
        torch.save(model.state_dict(), 'model_state.pt')
        # These functions only evaluate/display the publicly fitted model.
        for plot_or_metric in (toy.training_plots, toy.panels, toy.log_score,
                               toy.metrics, toy.diagnostics):
            plot_or_metric(data)
        table = metric_table('toy', data)
        table.to_csv('metrics.csv', index=False)
    print(table.to_string(index=False))
    return data


if __name__ == '__main__':
    reproduce(Path(__file__).resolve().parents[1] / 'results' / 'toy_public_api')
