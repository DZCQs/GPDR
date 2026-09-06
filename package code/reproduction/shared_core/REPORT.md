# Public GPDR Interface Reproduction

This audit supersedes the earlier per-example algorithm-port audit. All three
examples now construct `gpdr.GPDR`, call `gpdr.fit_adam`, and use that model's
`predict_density` (and, for toy panels, `predict_f_and_derivs`). These are the
same public functions used for new data. No example-specific subclass is used.

## Architecture

- The sole variational objective and density algorithm are in `gpdr/models.py`.
- The sole Adam/iterate-averaging loop is in `gpdr/training.py`.
- Derivative kernels and inducing-point utilities are shared.
- `gpdr/paper/configuration.py` only supplies data, base distributions, grids,
  hyperparameters and explicit numerical settings to the public interface.
- `paper/toy_model.py`, `weather_model.py`, `gini_model.py`, and their duplicated
  supporting code were removed. Original notebooks and data were not changed.
- `examples/toy_public_api.py` exposes the full toy reproduction without calling
  `run_example` or loading notebook code, saved parameters, or expected metrics.

## Verification Status

All three full public-core runs passed on 2026-09-06. Each started from the
original data at the selected hyperparameters, without loading fitted parameters.

| Example | Training steps | Required numeric records | Identical PNGs | Metric exports |
| --- | ---: | ---: | ---: | --- |
| Toy | 300,000 | 603 | 6 / 6 | Exact |
| Weather (MPS) | 5,000 | 3,070 | 11 / 11 | Exact |
| Gini | 2,000 | 925 | 5 / 5 | Exact |

All 4,598 required scientific records match by shape, dtype and SHA-256 of
their unrounded numeric values. All 22 rendered PNG files match byte-for-byte.
No required record or figure is missing. CSV/JSON metrics also match exactly.
See [comparison.json](comparison.json) for the machine-readable result.

| Example | GPDR summed log score | RMSE | 95% coverage | Average width |
| --- | ---: | ---: | ---: | ---: |
| Toy | 27910.20316205508 | 0.1615085560398703 | 0.9533 | 0.6061224049207613 |
| Weather | -10920.253705775578 | 4.035515462114401 | 0.9548190975740597 | 15.043455863604885 |
| Gini | 303.7592583122683 | 0.0688678867620055 | 0.9333333333333333 | 0.2439380730730731 |

The exported tables also contain original-base metrics and per-observation log
scores. The numeric audit includes the other interval levels, all plotted
curves, spatial summaries and residual diagnostics, not just this table.

Short comparisons against classes read from the untouched source notebooks passed
for toy, Gini, weather CPU and weather MPS: cached kernel/base matrices,
initialization, objective values and gradients, eight Adam updates, another
12 updates through the public `fit_adam` (including averaging/traces), and query
densities match exactly.
These checks are supplementary, not substitutes for full-run evaluation.

Public-interface unit tests also fit new three-covariate data with a nonlinear
Gaussian base, use callable beta bases, change covariate dimension, and confirm
that supplying a reference g0 does not change h. The final suite has 35 passing
tests; three opt-in historical-pickle tests are skipped. The built 0.3.0 wheel
was imported outside the source directory and passed the new-data, callable-base
and checkpoint smoke checks. It contains no removed per-example model modules.

`examples/toy_public_api.py` also ran through all its plotting/metric stages in
a separate 80-train/40-test/12-step smoke test. This checks the standalone entry
point; the full-size evidence is the 300,000-step run above through the same
public model and trainer. Source notebook hashes were rechecked unchanged.

## Comparison Contract

The reference is the prior independent **notebook** run, not the prior package
implementation. Source hashes are checked before comparing. All reference
scientific numeric records and figure pixels must match. Completion records are
bound to the verified source notebook hashes, and expected figure inventories
are checked against the numeric audit. A closed list of
optimizer-loop locals and unused GPyTorch wrapper parameters is excluded because
those implementation details no longer exist in the reusable optimizer/model.
The complete excluded-key list is written to the comparison JSON. Trained
variational parameters, the full recorded optimization traces, input data,
inducing points, kernel matrices, prediction arrays, diagnostics and metrics
remain required; no tolerance or rounded-value substitution is used.

Numerical equality applies to the recorded software/device configuration.
Wall-clock timing and PDF creation timestamps are not compared. No CV/BO or R
benchmark reruns are part of this verification.

The final source hashes are in [package_source_hashes.json](package_source_hashes.json).
Run logs and the corresponding audit, completion marker, metrics and figures
are under each `toy/package`, `weather/package` and `gini/package` directory.
The lightweight checkpoints retain trained parameters, kernel settings and the
verified inducing coordinates, not the large regenerable training caches.
