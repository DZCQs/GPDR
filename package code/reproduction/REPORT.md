# GPDR Reproduction Report

**Superseded:** this report tested the earlier per-example algorithm port. It
does not establish reproduction through a reusable public model. See
[the public-core audit](shared_core/REPORT.md) for the corrected implementation
and verification. The independent notebook reference artifacts below are kept
as the comparison baseline, not as evidence for the removed package code.

All three current GPDR workflows were trained and evaluated from scratch using
the package, and compared with independent executions of the original notebook
GPDR cells. No saved notebook model parameters or metric constants were injected
into the package runs. Original notebooks and input data were not edited.

## Exact Comparison

| Example | Numerical audit records | Identical rendered figures | Metric exports | Result |
| --- | ---: | ---: | --- | --- |
| Toy | 680 | 6 / 6 | CSV and JSON verified | Exact match |
| Weather | 3,243 | 11 / 11 | CSV and JSON verified | Exact match |
| Gini | 967 | 5 / 5 | CSV and JSON verified | Exact match |

There are no differing or missing numerical records. Checks cover data and split
values, inducing points, model parameters, training traces, evaluation arrays,
diagnostics, plot coordinates and rendered PNG bytes. Records include snapshots
at successive stages, not just distinct final scalar metrics. The PNG comparison
uses a common 90-dpi rendering setup; PDF timestamps are not compared.

Machine-readable evidence: [comparison.json](comparison.json). Each example's
`package` and `reference` directories contain `audit.json`, `completion.json`,
`environment.json`, `metrics.csv`, `metrics.json`, figures, and `run.log`.

## Reproduced GPDR Metrics

| Example | Test observations | Sum log score | Mean log score | RMSE | 95% coverage | Average 95% width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy | 50,000 | 27910.203162 | 0.558204 | 0.161509 | 0.953300 | 0.606122 |
| Weather | 16,777 | -10920.253706 | -0.650906 | 4.035515 | 0.954819 | 15.043456 |
| Gini | 210 | 303.759258 | 1.446473 | 0.068868 | 0.933333 | 0.243938 |

Toy's 50%, 80%, 90%, and 95% interval calculations all match, as do its Panel A-C
figures and residual diagnostic. Weather's temporal/spatial figures and its
4,000-observation residual diagnostic match. Gini's Monte Carlo metrics, density
comparisons, mean/residual plots, and quantile-residual diagnostic match.

## Conditions and Scope

- Toy: CPU/float64, 300,000 steps. Weather: Apple MPS/float32, 5,000 steps.
  Gini: CPU/float64, 2,000 steps. No reduced-step or reduced-test-set substitute
  was used for the full reproduction checks.
- Exact equality is established on the recorded software/device configuration.
  Dependency versions are pinned in `../requirements-reproduction.txt`.
  Wall-clock timings vary and are excluded from equality checks.
- Selected hyperparameters, RNG order, numerical grids, base-model construction,
  and existing plotting conventions are preserved. Completed CV/BO searches and
  the separate R benchmark cells were not rerun or changed.
- Unit tests: 18 passed; 3 optional historical-pickle tests skipped. These skipped
  tests concern external old artifacts, not the three current full-run audits.
- The built wheel was independently imported from outside the source directory;
  its toy preparation, fitting, and prediction smoke test passed.

See [README](../README.md) for installation and reproduction commands.
