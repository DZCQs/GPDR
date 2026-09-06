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

`prepare_gini_data` and `prepare_weather_data` likewise prepare their original
data and fitted bases. `gpdr.paper.configuration.gini_model_inputs` and
`weather_model_inputs` return explicit constructor arguments for **the same**
`GPDR(**inputs)`. `GINI_FIT` and `WEATHER_FIT` contain the arguments for the same
`fit_adam(model, **settings)`. Inspect these short configuration functions to
see the base/grid choices. They contain no variational objective or optimizer.

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
and soft Gumbel-softmax imputation with temperature 0.5. The configuration keeps
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

## Verification

The current public-core reproduction is documented in
[reproduction/shared_core/REPORT.md](reproduction/shared_core/REPORT.md).
All three full runs passed: 4,598 scientific numeric records, all metric exports,
and 22 rendered figures exactly match the independent notebook executions in
the recorded environment. The unit suite has 35 passing tests, with three
historical-pickle checks intentionally opt-in. The built wheel was also tested
on new data outside the source directory.
The earlier `reproduction/REPORT.md` concerns the superseded per-example port,
not proof of the public interface. Its independent notebook reference runs are
retained for comparison with the new full public-core runs.

```bash
python -m pytest tests -q
python scripts/check_shared_core.py --example toy --project-root ..
python scripts/check_shared_core.py --example weather --device mps --project-root ..
python scripts/check_shared_core.py --example gini --project-root ..
python scripts/compare_shared_core.py
```

The short tests read classes from untouched notebooks and compare initialization,
updates and predictions. Full-run audits use the public `GPDR` and `fit_adam`.
Comparison requires all reference scientific records and rendered figures to
match; optimizer-local loop variables and unused GPyTorch parameters are listed
separately, not counted as results. Source notebook hashes are checked first.
Unit tests also verify a new nonlinear three-covariate base through the same
public interface and reject reintroduction of per-example GPDR implementations.

To rerun a complete audit:

```bash
python scripts/verify_paper.py --example gini --mode package --project-root .. --output reproduction/shared_core/gini/package
```

Rebuild the model with identical data/base/configuration and pass the saved
`state_dict['Vu']` as `inducing_points` before calling `load_state_dict`.
Loading rejects mismatched inducing locations or kernel settings rather than
silently using different cached matrices. Full-model pickling supports functional bases via cloudpickle,
but is environment-dependent. Only load pickle/checkpoint files you trust.
The earlier multix tutorial remains a legacy experiment using deterministic
imputation, not the reference for these three updated paper examples.
