# GPDR

Gaussian process density regression: [paper](https://arxiv.org/abs/2606.22915).

## Contents

- [Simulations](Simulations): the simulation study notebook, 100 hyperparameter candidates, and 200 CV/BO results.
- [Weatherdata](Weatherdata): the weather example notebook, candidates, CV/BO results, and input data.
- [Gini](Gini): the Gini example notebook, candidates, CV/BO results, and input data.
- [package code](package%20code): the reusable Python package, public API example, tests, and reproduction scripts.

## Run the Notebooks

Open each notebook within its own folder as the working directory. For weather example,
extract `weather_data.csv` from `Weatherdata/weather_data.csv.zip` into
`Weatherdata` first.

The commented Part 1 / Part 2 cells retain the candidate generation and CV/BO
procedures. Final search results are in `hyperparameter_search_cv_bo_results_toy.pkl`,
`hyperparameter_search_cv_results_weather.pkl`, and
`hyperparameter_search_cv_results_withbo_gini.pkl`, respectively.

## Use the Package

See the [package README](package%20code/README.md) for installation, the generic
`GPDR` / `fit_adam` / `predict_density` interface, and reproduction settings.

