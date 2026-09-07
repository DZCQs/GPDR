# Logistic Gaussian process density regression: a generalized Bayesian approach

Density regression extends conventional parametric regression by allowing the entire distribution of the response to vary flexibly with covariates rather than just low-order moments. In the Bayesian setting, logistic Gaussian process (GP) priors have been widely used for density estimation and extend naturally to density regression. The prior can be centred on a base density model, with the nonparametric component providing an interpretable correction that is useful for model criticism. However, logistic GP density regression models have seen limited use, since they require computation of a normalizing constant for every observation, typically via numerical integration. We address this difficulty by proposing a generalized Bayesian approach using a loss function based on the Hyvarinen score. The Hyvarinen score depends only on derivatives of the log density with respect to the response, eliminating the need to compute normalizing constants. Since GP computations remain expensive, we also employ sparse inducing point approximations and variational inference to develop a scalable approach. We demonstrate the method on one simulated and two real datasets, including a German weather dataset with more than 150,000 observations.

The preprint can be found at https://doi.org/10.48550/arXiv.2606.22915.

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

## Citation

If you find this code helpful, please cite our work

```bibtex
@article{CheKocLeeNot2026,
  title={Logistic Gaussian process density regression: a generalized Bayesian approach},
  author={Chen, Zichuan and Kock, Lucas and Lee, Jeong Eun and Nott, David J},
  journal={arXiv preprint arXiv:2606.22915},
  year={2026}
}
```

