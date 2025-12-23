+++
title = "Release 0.8.1"
date = "2025-12-23"
+++

`linfa 0.8.1` adds random forest, and AdaBoost boosting algorithms in `linfa-ensemble` and least angle regression in a new `linfa-lars` crate.

<!-- more -->

## Improvements and fixes

* `linfa-ica`: fix missing exponential by @lmmx in https://github.com/rust-ml/linfa/pull/426
* `linfa`:
  * add bootstrap-with-indices utilities for `Dataset` 
  * fix ndarray 0.16/0.17 versions mismatch (0.17 not supported) 

## New algorithms

### Least Angle regression

Least Angle Regression (LARS) is an algorithm used in regression for high dimensional data.

See [sklearn.linear_model](https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression)

### Ensemble methods

Ensemble methods combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.

#### Random Forest

See [sklearn.ensemble forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)

#### AdaBoost 

See [sklearn.ensemble AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
