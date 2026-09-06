# GPDR

Gaussian process density regression: [paper](https://arxiv.org/abs/2606.22915).

## Contents

- [Simulations](Simulations): the final toy notebook, 100 hyperparameter candidates, and 200 CV/BO results.
- [Weatherdata](Weatherdata): the final weather notebook, candidates, CV/BO results, and input data.
- [Gini](Gini): the final Gini notebook, candidates, CV/BO results, and input data.
- [package code](package%20code): the reusable Python package, public API example, tests, and reproduction scripts.

Only the final `new_benchmark` / `new_benchmarks` notebooks are included. Their
saved outputs are retained. Generated figures and fitted model files are not
uploaded separately.

## Run the Notebooks

Open each notebook with its own folder as the working directory. For weather,
extract `weather_data.csv` from `Weatherdata/weather_data.csv.zip` into
`Weatherdata` first. On macOS or Linux, from the repository root:

```bash
unzip Weatherdata/weather_data.csv.zip weather_data.csv -d Weatherdata
```

The commented Part 1 / Part 2 cells retain the candidate generation and CV/BO
procedures. Final search results are in `hyperparameter_search_cv_bo_results_toy.pkl`,
`hyperparameter_search_cv_results_weather.pkl`, and
`hyperparameter_search_cv_results_withbo_gini.pkl`, respectively.

## Use the Package

See the [package README](package%20code/README.md) for installation, the generic
`GPDR` / `fit_adam` / `predict_density` interface, and reproduction settings.
The existing repository folder names are preserved. The package's paper data
helpers use the original local folder names. On macOS or Linux, create these
aliases from the repository root before using the paper helpers:

```bash
ln -s Simulations toyexample_thickertail
ln -s Weatherdata weatherdata_thickertail
ln -s Gini giniindex_thickertail
cd 'package code'
python -m pip install -r requirements-reproduction.txt
python -m pip install -e '.[examples,test]'
```

Alternatively, rename the three folders to the corresponding local names in
your downloaded copy. This changes paths only, not the data or computations.
