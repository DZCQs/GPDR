# GPDR

This is a package version of the GP density-regression code from the GPDR paper notebooks.
It keeps the notebook hyperparameters and data-preparation choices available as example
runners, while moving the reusable pieces into importable modules.

## Install

```bash
pip install -e .
```

For the weather example install the optional dependencies:

```bash
pip install -e ".[examples]"
```

## Quick Checks

From this folder:

```bash
python -m pytest tests
python scripts/reproduce_examples.py --example toy --mode saved
python scripts/reproduce_examples.py --example gini --mode saved
python scripts/reproduce_examples.py --example weather --mode saved
```

The `--mode saved` path loads the existing notebook-generated pickle and evaluates the
same variational objective from the package classes.

## Full Reproduction Runs

These retrain from the notebook setup and compare against the metric values stored in
the notebooks:

```bash
python scripts/train_and_compare.py --example toy --steps 400000 --M 1000
python scripts/train_and_compare.py --example gini --steps 2000 --M 1000 --n-samples 2000
python scripts/train_and_compare.py --example weather --steps 5000 --M 1000
```

The weather notebook uses accelerator-aware float32 code. CPU retraining is slower, but
the package preserves the notebook's multi-dimensional normalization convention.

## Python API Sketch

```python
from gpdr.examples import load_toy_saved

bundle = load_toy_saved("/Users/zichuanchen/Desktop/gp_density_regression")
model = bundle.model
objective, info = model.forward_objective()
```

The original notebook folders are used as read-only data/artifact inputs.
