# GPDR

Gaussian process density regression with a reusable conditional-base interface.
The same `GPDR` class, `fit_adam` optimizer, and `predict_density` implementation
are used by the toy, German weather, and Gini examples and by new datasets.
There are no per-example copies of the GPDR objective or prediction algorithm.

## Install

From this directory, in the Python environment used by your notebook:

```bash
python -m pip install -r requirements-reproduction.txt
python -m pip install -e '.[examples,test]'
```

The pinned reproduction environment is Python 3.13.2 on macOS arm64. Exact
numerical equality requires the recorded software stack and device. Toy/Gini
use CPU/float64; the recorded weather run uses Apple MPS/float32. Timing is
measured afresh and is not expected to match.

## The Public Interface

```python
from gpdr import GPDR, KernelParams, fit_adam, sample_inducing

# x_train: floating tensor (n, d); y_train: floating tensor (n,).
# base: your fitted conditional distribution, with the methods described below.
model = GPDR(
    x_train, y_train,
    base=base,
    inducing_points=lambda V: sample_inducing(V, count=200, seed=2),
    kp=kernel_parameters,
    C=2,
    beta=learning_weight,
)
model, trace = fit_adam(model, steps=2000, lr=0.5)
prediction = model.predict_density(x_test[0], y_min=-3, y_max=3, M=1000)

y_grid = prediction.y
h_hat = prediction.density
```

This interface does not take an example name. Data generation, train/test
splitting, covariate transformation S, fitting a base, and selecting
hyperparameters are the caller's decisions. Supply S(x) in `x_train` and the
same coordinates at prediction time. These operations are not hidden in GPDR.

`DensityPrediction` has named fields: `y`, `base_density`, `reference_density`,
`density`, `correction`, and `pit`. `correction` is exp(f), aligned with the
response grid; it is not the density ratio relative to an original g0.
An optional `reference_base` supplies g0 for comparison only. It never changes h.
`model.predict_f_and_derivs(...)` returns PIT, f, f_z and f_zz on a response grid.

### Supply a New Base Model

Your base must implement:

```python
base.cdf(y, x)     # G(y | x)
base.pdf(y, x)     # g(y | x)
base.terms(y, x)   # (g, derivative of g in y, second derivative of g in y)
```

Each output must have y's shape, dtype and device. The package supplies
`BetaBase`, `StudentTLinearBase`, `StudentTLinearIndexBase`, `StudentTGamBase`,
and a function adapter `FunctionalBase`. For example, a fitted nonlinear mean
and a constant Gaussian scale can be connected without changing any GPDR code:

```python
import math
import torch
from gpdr import FunctionalBase

def pdf(y, x):
    mu = fitted_mean(x)
    return torch.exp(-0.5 * ((y - mu) / scale)**2) / (math.sqrt(2*math.pi) * scale)

def cdf(y, x):
    return torch.distributions.Normal(fitted_mean(x), scale).cdf(y)

def terms(y, x):
    g = pdf(y, x)
    score = -(y - fitted_mean(x)) / scale**2
    return g, g * score, g * (score**2 - 1 / scale**2)

base = FunctionalBase(cdf=cdf, pdf=pdf, terms=terms)
```

Use the desired diffuse/heavy-tailed base here; GPDR does not silently alter it.
For beta regression `BetaBase(mu_function, phi)` supports a callable conditional
mean. If base covariates differ from GP covariates, another option is to supply
the already evaluated conditional base at prediction time:
`model.predict_density(x_star, ..., base=BetaBase(mu_star, phi))`.

Inducing values are explicit tensors in joint (S(x), z) coordinates, or a
callable receiving the training (S(x), PIT) rows. `tensor_grid`,
`sample_inducing`, and `kmeans_inducing` are reusable constructors. Neither
coordinates nor response support are limited to a particular example.

## Reproduce the Toy with Public Functions

This is the actual supported model, not a toy-specific GPDR class:

```python
import math
import numpy as np
import torch
from gpdr import GPDR, KernelParams, fit_adam, prepare_toy_data, tensor_grid

data = prepare_toy_data()  # identical generated observations, split, and fitted base
x, y = data['x'], data['y']
torch.manual_seed(2)
np.random.seed(2)

model = GPDR(
    x, y, base=data['base'], C=2, beta=1031.17,
    inducing_points=tensor_grid(
        torch.linspace(x.min(), x.max(), 12),
        torch.linspace(-0.05, 1.05, 12),
    ),
    kp=KernelParams(
        log_sigma2=math.log(0.9199**2),
        log_lx2=math.log(0.0786**2),
        log_lz2=math.log(0.3363**2),
    ),
)
model, trace = fit_adam(
    model, steps=300000, lr=0.5, average_last=5000,
    averaging='running', trace_at='before', verbose_every=10000,
)
prediction = model.predict_density(0.3, y_min=-0.3, y_max=3.0, M=2000)
```

The complete runnable version, including all original plots and metrics, is
[examples/toy_public_api.py](examples/toy_public_api.py):

```bash
python examples/toy_public_api.py
```

## Reproduce Weather with Public Functions

Run this block from the package directory in the recorded environment. The
parent directory must contain `Weatherdata` (the GitHub layout) or
`weatherdata_thickertail` (the original local layout). Extract
`weather_data.csv.zip` there if needed, and keep the `de.shp` shapefile and its
companion files together for the maps.

The preparation functions reproduce the data transformations, split and fitted
base. The model, inducing points, hyperparameters and optimizer are specified
explicitly below using the public API. The final functions only evaluate and
plot this fitted model; they do not train another model or read notebook outputs.

```python
import math
from pathlib import Path

import torch
from gpdr import GPDR, FunctionalBase, KernelParams, fit_adam, kmeans_inducing
from gpdr.paper import weather
from gpdr.paper.results import metric_table
from gpdr.paper.runner import output_directory

project_root = Path('..').resolve()
data_dir = project_root / 'Weatherdata'
if not data_dir.is_dir():
    data_dir = project_root / 'weatherdata_thickertail'
data = {'data_dir': data_dir}

with output_directory('results/weather_public_api'):
    # Preserve the notebook's preparation and plotting order.
    for prepare in (
        weather.setup, weather.seed, weather.prepare, weather.split,
        weather.covariate_histogram, weather.fit_base, weather.base_scale,
        weather.pit_scatter, weather.base_histogram, weather.base_summary,
        weather.map_setup, weather.base_effects, weather.helpers,
    ):
        prepare(data)

    device = torch.device('mps')  # Device used for the reported weather run.
    torch.set_default_dtype(torch.float32)
    for name in ('x_train', 'y_train', 'x_test', 'y_test'):
        data[name] = torch.tensor(data[name], dtype=torch.float32, device=device)
    x, y = data['x_train'], data['y_train']

    base_terms = data['base_g_terms']
    base = FunctionalBase(
        cdf=data['to_z_from_y'], terms=base_terms,
        pdf=lambda y, x: base_terms(y, x)[0],
    )
    gam, sigma = data['gam'], data['ORIGINAL_SIGMA_NORMAL']

    def original_pdf(y, x):
        mu = torch.tensor(gam.predict(x.cpu().numpy()),
                          device=x.device, dtype=x.dtype)
        return (1.0 / (math.sqrt(2.0 * math.pi) * sigma)
                * torch.exp(-0.5 * ((y - mu) / sigma) ** 2))

    reference_base = FunctionalBase(cdf=None, terms=None, pdf=original_pdf)
    m_induce = 1000
    per_axis = int(round(m_induce ** (1 / (x.shape[1] + 1))))
    model = GPDR(
        x, y, base=base, reference_base=reference_base,
        inducing_points=lambda V: kmeans_inducing(V, m_induce)[per_axis:],
        C=2, beta=29.6,
        kp=KernelParams(
            log_sigma2=math.log(0.1806**2),
            log_lx2=torch.tensor([math.log(0.4901**2)] * x.shape[1],
                                dtype=x.dtype, device=x.device),
            log_lz2=math.log(0.236**2),
        ),
        normalization_size=x.numel(), loss_reduction='sum',
        warm_start_solver='solve', exp_clip=None,
    ).to(device)
    model, trace = fit_adam(
        model, steps=5000, lr=0.5, average_last=500,
        averaging='sum', trace_at='after', verbose_every=500,
    )
    trace = {key: trace[key] for key in ('F', 'w', 'mu_norm', 's_component_mean')}
    trace['s_mean'] = trace.pop('s_component_mean')
    data.update(model=model, trace=trace)

    for evaluate in (
        weather.density_plots, weather.log_score, weather.prediction_helpers,
        weather.time_effects, weather.spatial_effects, weather.spatial_metrics,
        weather.metrics, weather.diagnostics,
    ):
        evaluate(data)
    table = metric_table('weather', data)
    table.to_csv('metrics.csv', index=False)
    torch.save(model.state_dict(), 'model_state.pt')
    print(table.to_string(index=False))

# A direct prediction from the same publicly trained model.
prediction = model.predict_density(
    data['x_test'][0], y_min=data['y_all'].min(),
    y_max=data['y_all'].max(), M=1000,
)
y_grid = prediction.y * data['y_std'] + data['y_mean']
h_hat = prediction.density / data['y_std']  # Density on the Celsius scale.
```

The inducing construction intentionally retains the notebook's omitted first
covariate center, giving 1,290 inducing points. The Student-t training base and
the original Gaussian reference are separate. For exact reproduction, keep the
MPS/float32 computation, score scaling and parameter averaging shown above.

## Reproduce Gini with Public Functions

Run this block from the package directory. The parent directory must contain
`Gini/reg_df.csv` (the GitHub layout) or `giniindex_thickertail/reg_df.csv` (the
original local layout). As above, data/base preparation is example-specific;
the GPDR model, training and prediction use the same public functions.

```python
import math
from pathlib import Path

import torch
from gpdr import GPDR, BetaBase, KernelParams, fit_adam, sample_inducing
from gpdr.paper import gini
from gpdr.paper.results import metric_table
from gpdr.paper.runner import output_directory

project_root = Path('..').resolve()
data_dir = project_root / 'Gini'
if not data_dir.is_dir():
    data_dir = project_root / 'giniindex_thickertail'
data = {'data_dir': data_dir}

with output_directory('results/gini_public_api'):
    for prepare in (
        gini.setup, gini.load_data, gini.data_summary, gini.split,
        gini.fit_base, gini.coefficients, gini.base_means,
        gini.base_plots, gini.prepare,
    ):
        prepare(data)

    # Preparation sets CPU/float64 and both RNG seeds to 0, as in the notebook.
    x, y = data['x_t'], data['y_t']
    base = BetaBase(data['mu_base_train'], data['phi_base'])
    model = GPDR(
        x, y, base=base,
        inducing_points=lambda V: sample_inducing(V, count=200, seed=2),
        C=2, beta=493.61, train_pit_epsilon=1e-6,
        kp=KernelParams(
            log_sigma2=math.log(0.5769**2),
            log_lx2=torch.log(torch.full((x.shape[1],), 0.507**2,
                                        dtype=x.dtype, device=x.device)),
            log_lz2=math.log(0.8355**2),
        ),
    ).to('cpu')
    model, trace = fit_adam(
        model, steps=2000, lr=0.5, trace_at='after', verbose_every=200,
    )
    trace = {key: trace[key] for key in ('F', 'w', 's_mean')}
    data.update(model=model, trace=trace)

    # Keep the original order, including the Monte Carlo evaluation draws.
    for evaluate in (
        gini.density_plots, gini.mc_means, gini.residual_plot,
        gini.log_score, gini.metrics, gini.diagnostics,
    ):
        evaluate(data)
    table = metric_table('gini', data)
    table.to_csv('metrics.csv', index=False)
    torch.save(model.state_dict(), 'model_state.pt')
    print(table.to_string(index=False))

# Evaluate the fitted conditional base at a new row, then predict with GPDR.
row = data['df_test'].iloc[0]
x_star = (row[data['gp_cols']].to_numpy(dtype=float) - data['x_min'].to_numpy())
x_star = x_star / (data['x_max'].to_numpy() - data['x_min'].to_numpy() + 1e-12)
design_row = torch.from_numpy(data['X_test'].iloc[0].to_numpy(dtype=float))
mu_star = torch.sigmoid(torch.from_numpy(data['beta_hat']) @ design_row)
prediction = model.predict_density(
    x_star, y_min=0.001, y_max=0.999, M=1000,
    base=BetaBase(mu_star, data['phi_base']),
    reference_base=BetaBase(mu_star, data['phi_base_original']),
)
y_grid, h_hat = prediction.y, prediction.density
```

`phi_base` is half the fitted original precision and is used for GPDR training
and prediction. `phi_base_original` is used only for the reference comparison.
The evaluation functions retain the notebook's grids and Monte Carlo rules so
the figures and metrics come from this trained model using the same procedure.

## All Paper Figures and Metrics

Convenience orchestration is still available, but it calls the public functions
above. From gpdr_package with the original data folders in its parent:

```bash
python -m gpdr.paper --example toy --project-root .. --output results/toy
python -m gpdr.paper --example weather --project-root .. --output results/weather
python -m gpdr.paper --example gini --project-root .. --output results/gini
```

Or, in Jupyter:

```python
from gpdr import run_example
result = run_example('gini', project_root='..', output_dir='results/gini')
result['metric_table']
```

Each run trains from scratch at the selected hyperparameters. It writes PDF
figures, `model_state.pt`, `metrics.csv` and `metrics.json`. Jupyter displays the
plots; the CLI also saves displayed figures as PNG. The table includes summed
and per-test-observation log scores. No saved notebook model or hard-coded
metric is used to generate results. Original notebooks/data are never written.
CV/BO searches and the separate R benchmarks are not run.

External inputs are `weatherdata_thickertail/weather_data.csv`, the existing
German map shapefile and companion files, and `giniindex_thickertail/reg_df.csv`.
Run examples sequentially or in separate processes because RNGs and plotting
settings are process-global. Do not change cell order or insert random draws
if reproducing a notebook's stochastic evaluation.

| Setting | Toy | Weather | Gini |
| --- | --- | --- | --- |
| Train/test observations | 5,000 / 50,000 | 150,985 / 16,777 | 1,883 / 210 |
| Mixture components | 2 | 2 | 2 |
| Actual inducing points | 144 | 1,290 | 200 |
| Adam learning rate | 0.5 | 0.5 | 0.5 |
| Steps | 300,000 | 5,000 | 2,000 |
| Parameter averaging | last 5,000, running mean | last 500, sum/divide | none |

The three models share trainable mixture weights, Goldberger entropy correction,
and soft Gumbel-softmax imputation with temperature 0.5. There are `C-1` trainable
logits; a fixed zero reference logit is appended before softmax to obtain all `C`
weights, which sum to one. Initialization draws `C` values and subtracts the last
one, preserving the notebooks' initial weights and random-number sequence.
The configuration keeps
the existing notebook's numerical choices explicit: weather uses `x.numel()`
for score scaling, a direct warm-start solve, an omitted first k-means center,
and unclipped exponentiation. Toy's true SD is `0.2*x + 0.05`. Gini's training
base precision is halved and its reported metrics use the original interleaved
3,000-draw Monte Carlo evaluation. Existing plot conventions, including Gini's
five-point omission/jitter in the residual plot, are retained, not reinterpreted.

For a new dataset, `evaluate_predictions(y_true, predictions)` computes the four
metrics from public prediction objects by numerical integration. Use the paper
evaluation recipes when reproducing their specific grids, unit conversions and
Monte Carlo conventions. A base/reference density must stay on the same response
scale as the observations to compare log scores.

## Save and Load

Rebuild the model with identical data/base/configuration and pass the saved
`state_dict['Vu']` as `inducing_points` before calling `load_state_dict`.
Loading rejects mismatched inducing locations or kernel settings rather than
silently using different cached matrices. Older model checkpoints with `C` logits
are converted to `C-1` reference logits on load, preserving their mixture weights
up to floating-point rounding. Old Adam optimizer states are not converted.
Full-model pickling supports functional bases via cloudpickle,
but is environment-dependent. Only load pickle/checkpoint files you trust.
