+++
title = "Release 0.5.0"
date = "2021-10-20"
+++

Linfa's 0.5.0 release adds initial support for the OPTICS algorithm, multinomials logistic regression, and the family of nearest neighbor algorithms. Furthermore, we have improved documentation and introduced hyperparameter checking to all algorithms.

<!-- more -->

## New algorithms

[OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm) is an algorithm for finding density-based clusters. It can produce reachability-plots, hierarchical structure of clusters. Analysing data without prior assumption of any distribution is a common use-case. The algorithm is added to `linfa-clustering` and an example can be find at [linfa-clustering/examples/optics.rs](https://github.com/rust-ml/linfa/blob/master/algorithms/linfa-clustering/examples/optics.rs).

Extending logistic regression to the multinomial distribution generalizes it to [multiclass problems](https://en.wikipedia.org/wiki/Multinomial_logistic_regression). This release adds support for multinomial logistic regression to `linfa-logistic`, you can experiment with the example at [linfa-logistic/examples/winequality_multi.rs](https://github.com/rust-ml/linfa/blob/master/algorithms/linfa-logistic/examples/winequality_multi.rs).

Nearest neighbor search finds the set of neighborhood points to a given sample. It appears in numerous fields of applications as a distance metric provider. (e.g. clustering) This release adds a family of nearest neighbor algorithms, namely [Ball tree](https://en.wikipedia.org/wiki/Ball_tree), [K-d tree](https://en.wikipedia.org/wiki/K-d_tree) and naive linear search. You can find an example in the next section.

## Improvements

 * use least-square solver from `ndarray-linalg` in `linfa-linear`
 * make clustering algorithms generic over distance metrics
 * bump `ndarray` to 0.15
 * introduce `ParamGuard` trait for explicit and implicit parameter checking (read more in the [CONTRIBUTE.md](https://github.com/rust-ml/linfa/blob/master/CONTRIBUTE.md#parameters-and-checking))
 * improve documentation in various places

## Nearest Neighbors

You can now choose from a growing list of NN implementations. The family provides efficient distance metrics to KMeans, DBSCAN etc. The example shows how to use KDTree nearest neighbor to find all the points in a set of observations that are within a certain range of a candidate point.

You can query nearest points explicitly:

```rust
// create a KDTree index consisting of all the points in the observations, using Euclidean distance
let kdtree = CommonNearestNeighbour::KdTree.from_batch(observations, L2Dist)?;
let candidate = observations.row(2);
let points = kdtree.within_range(candidate.view(), range)?;
```

Or use one of the distance metrics implicitly, here demonstrated for KMeans:

```rust
use linfa_nn::distance::LInfDist;

let model = KMeans::params_with(3, rng, LInfDist)
    .max_n_iterations(200)
    .tolerance(1e-5)
    .fit(&dataset)?;
```

