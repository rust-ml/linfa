+++
title = "Release 0.6.0"
date = "2022-06-15"
+++

Linfa's 0.6.1 release mainly consists of fixes to existing algorithms and the overall crate. The Isotonic Regression algorithm has also been added to `linfa-linear`.

## Improvements and fixes

 * Add constructor for `LpDist` in `linfa-nn`.
 * Add `Send + Sync` to trait objects returned by `linfa-nn`, which are now aliased as `NearestNeighbourBox`.
 * Remove `anyhow <= 1.0.48` version restriction from `linfa`.
 * Bump `ndarray` dependency to 0.15
 * Fix `serde` support for `LogisticRegression` in `linfa-logistic`.

## New algorithms

Isotonic regression fits a free-form line to the training data. Unlike linear regression, which fits a straight line, isotonic regression can result in a much closer fit to the data. The algorithm has been added to `linfa-linear`.

Mean absolution percentage error (MAPE) is a method of measuring the difference between two datasets, and has been added to the main `linfa` crate.
