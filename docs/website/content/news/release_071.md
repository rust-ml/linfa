+++
title = "Release 0.7.1"
date = "2025-01-14"
+++

Linfa's 0.7.1 release mainly consists of fixes to existing algorithms and the overall crate. The Random Projection algorithm has also been added to `linfa-reduction`.

<!-- more -->

## Improvements and fixes

 * add `serde` support to  `linfa-clustering`
 * add accessors for classes in `linfa-logistics` 
 * add accessors for `Pca` attributes in `linfa-reduction`
 * add `wasm-bindgen`feature to use linfa in the browser
 * fix covariance update for `GaussianMixtureModel` in `linfa-clustering`
 * bump `ndarray-linalg` to 0.16 and `argmin` to 0.9.0
 * bump MSRV to 1.71.1

## New algorithms

Random projections are a simple and computationally efficient way to reduce the dimensionality of the data by trading a controlled amount of accuracy (as additional variance) for faster processing times and smaller model sizes.

The dimensions and distribution of random projections matrices are controlled so as to preserve the pairwise distances between any two samples of the dataset.

See also [sklearn.random_projection](https://scikit-learn.org/stable/api/sklearn.random_projection.html)
