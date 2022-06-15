+++
title = "Release 0.6.0"
date = "2022-06-15"
+++

Linfa's 0.6.0 release removes the mandatory dependency on external BLAS libraries (such as `intel-mkl`) by using a pure-Rust linear algebra library. It also adds the Naive Multinomial Bayes and Follow The Regularized Leader algorithms. Additionally, the `AsTargets` trait has been separated into `AsSingleTargets` and `AsMultiTargets`.

<!-- more -->

## No more BLAS

With older versions of Linfa, algorithm crates that used advanced linear algebra routines needed to be linked against an external BLAS library such as Intel-MKL. This is done by adding feature flags like `linfa/intel-mkl-static` to the build, and it increased the compile times significantly. Version 0.6.0 replaces the BLAS library with a pure-Rust implementation of all the required routines, which Linfa uses by default. This means all Linfa crates now build properly and quickly without any extra feature flags. It is still possible for the affected algorithm crates to link against an external BLAS libary. Doing so requires enabling the crate's `blas` feature, along with the feature flag for the external BLAS library. The affected crates are as follows:

* `linfa-ica`
* `linfa-reduction`
* `linfa-clustering`
* `linfa-preprocessing`
* `linfa-pls`
* `linfa-linear`
* `linfa-elasticnet`

## New algorithms

Multinomial Naive Bayes is a family of Naive Bayes classifiers that assume independence between variables. The advantage is a linear fitting time with maximum-likelihood training in a closed form. The algorithm is added to `linfa-bayes` and an example can be found at [linfa-bayes/examples/winequality_multinomial.rs](https://github.com/rust-ml/linfa/blob/master/algorithms/linfa-bayes/examples/winequality_multinomial.rs).

[Follow The Regularized Leader (FTRL)](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf) is a linear model for CTR prediction in online learning settings. It is a special type of linear model with sigmoid function which uses L1 and L2 regularization. The algorithm is contained in the newly-added `linfa-ftrl` crate, and an example can be found at [linfa-ftrl/examples/winequality.rs](https://github.com/rust-ml/linfa/blob/master/algorithms/linfa-ftrl/examples/winequality.rs).

## Distinguish between single and multi-target

Version 0.6.0 introduces a major change to the `AsTarget` trait, which is now split into `AsSingleTargets` and `AsMultiTargets`. Additionally, the `Dataset*` types are parametrized by target dimensionality, instead of always using a 2D array. Furthermore, algorithms that work on single-target data will no longer accept multi-target datasets as input. This change may cause build errors in existing code that call the affected algorithms. The fix for it is as simple as adding `Ix1` to the end of the type parameters for the dataset being passed in, which forces the dataset to be single-target.

## Improvements

 * Remove `SeedableRng` trait bound from `KMeans` and `GaussianMixture`.
 * Replace uses of Isaac RNG with Xoshiro RNG.
 * `cross_validate` changed to `cross_validate_single`, which is for single-target data; `cross_validate_multi` changed to `cross_validate`, which is for both single and multi-target datasets.
 * The probability type `Pr` has been constrained to `0. <= prob <= 1.`. Also, the simple `Pr(x)` constructor has been replaced by `Pr::new(x)`, `Pr::new_unchecked(x)`, and `Pr::try_from(x)`, which ensure that the invariant for `Pr` is met.
