Unreleased
========================

Changes
----------------------
* remove `SeedableRng` trait bound from `KMeans` and `GaussianMixture`

Breaking Changes
----------------------
 * parametrize `AsTargets` by the dimensionality of the targets and introduce `AsSingleTargets` and `AsMultiTargets`
 * 1D target arrays are no longer converted to 2D when constructing `Dataset`s
 * `Dataset` and `DatasetView` can now be parametrized by target dimensionality, with 2D being the default
 * single-target algorithms no longer accept 2D target arrays as input
 * `cross_validate` changed to `cross_validate_single`
 * `cross_validate_multi` changed to `cross_validate`, which is now generic across single and multi-targets
 * `Pr` has been constrained to `0. <= prob <= 1.`. Creating probability value now possible with `Pr::new(prob)` or `Pr::try_from(prob)?`

Version 0.5.1 - 2022-02-28
========================

Note that the commits for this release are in the `0-5-1` branch.

Changes
-----------
 * remove `Float` trait bound from many `Dataset` impls, making non-float datasets usable
 * fix build errors in 0.5.0 caused by breaking minor releases from dependencies
 * fix bug in k-means where the termination condition of the algorithm was calculated incorrectly
 * fix build failure when building `linfa` alone, caused by incorrect feature selection for `ndarray`

Version 0.5.0 - 2021-10-20
========================

New Algorithms
-----------

 * Nearest neighbour algorithms and traits have been added as `linfa-nn` by [@YuhanLiin]
 * OPTICS has been added to `linfa-clustering` by [@xd009642]
 * Multinomial logistic regression has been added to `linfa-logistic` by [@YuhanLiin]

Changes
-----------
 * use least squares solver from `ndarray-linalg` in `linfa-linear` (3dc9cb0)
 * optimized DBSCAN by replacing linear range query implementation with KD-tree (44f91d0)
 * allow distance metrics other than Euclidean to be used for KMeans (4e58d8d)
 * enable models to write prediction results into existing memory without allocating (37bc25b)
 * bumped `ndarray` version to 0.15 and reduced duplicated dependencies (603f821)
 * introduce `ParamGuard` trait to algorithm parameter sets to enable both explicit and implicit parameter checking (01f912a)
 * replace uses of HNSW with `linfa-nn` (208a762)

Version 0.4.0 - 2021-04-28
========================

New Algorithms
-----------

 * Partial Least Squares Regression has been added as `linfa-pls` by [@relf]
 * Barnes-Hut t-SNE wrapper has been added as `linfa-tsne` by [@frjnn]
 * Count-vectorizer and IT-IDF normalization has been added as `linfa-preprocessing` by [@Sauro98]
 * Platt scaling has been added to `linfa-svm` by [@bytesnake]
 * Incremental KMeans and KMeans++ and KMeans|| initialization methods added to `linfa-clustering` by [@YuhanLiin]

Changes
-----------
 * bumped `ndarray` version to 0.14 (8276bdc)
 * change trait signature of `linfa::Fit` to return `Result` (a5a479f)
 * add `cross_validate` to perform K-folding (a5a479f)

Version 0.3.1 - 2021-03-11
========================

In this release of Linfa the documentation is extended, new examples are added and the functionality of datasets improved. No new algorithms were added.

The meta-issue [#82](https://github.com/rust-ml/linfa/issues/82) gives a good overview of the necessary documentation improvements and testing/documentation/examples were considerably extended in this release. 

Further new functionality was added to datasets and multi-target datasets are introduced. Bootstrapping is now possible for features and samples and you can cross-validate your model with k-folding. We polished various bits in the kernel machines and simplified the interface there.

The trait structure of regression metrics are simplified and the silhouette score introduced for easier testing of K-Means and other algorithms.

Changes
-----------
 * improve documentation in all algorithms, various commits
 * add a website to the infrastructure (c8acc785b)
 * add k-folding with and without copying (b0af80546f8)
 * add feature naming and pearson's cross correlation (71989627f)
 * improve ergonomics when handling kernels (1a7982b973)
 * improve TikZ generator in `linfa-trees` (9d71f603bbe)
 * introduce multi-target datasets (b231118629)
 * simplify regression metrics and add cluster metrics (d0363a1fa8ef)

Version 0.3.0 - 2021-01-21
=========================

New Algorithms
-----------

 * Approximated DBSCAN has been added to `linfa-clustering` by [@Sauro98]
 * Gaussian Naive Bayes  has been added to `linfa-bayes` by [@VasanthakumarV]
 * Elastic Net linear regression has been added to `linfa-elasticnet` by [@paulkoerbitz] and [@bytesnake]

Changes
----------

 * Added benchmark to gaussian mixture models (a3eede55)
 * Fixed bugs in linear decision trees, added generator for TiKZ trees (bfa5aebe7)
 * Implemented serde for all crates behind feature flag (4f0b63bb)
 * Implemented new backend features (7296c9ec4)
 * Introduced `linfa-datasets` for easier testing (3cec12b4f)
 * Rename `Dataset` to `DatasetBase` and introduce `Dataset` and `DatasetView` (21dd579cf)
 * Improve kernel tests and documentation (8e81a6d)

Version 0.2.0 - 2020-11-26
==========================

New algorithms
-----------

 - Ordinary Linear Regression has been added to `linfa-linear` by [@Nimpruda] and [@paulkoerbitz]
 - Generalized Linear Models has been added to `linfa-linear` by [VasanthakumarV]
 - Linear decision trees were added to `linfa-trees` by [@mossbanay]
 - Fast independent component analysis (ICA) has been added to `linfa-ica` by [@VasanthakumarV]
 - Principal Component Analysis and Diffusion Maps have been added to `linfa-reduction` by [@bytesnake]
 - Support Vector Machines has been added to `linfa-svm` by [@bytesnake]
 - Logistic regression has been added to `linfa-logistic` by [@paulkoerbitz]
 - Hierarchical agglomerative clustering has been added to `linfa-hierarchical` by [@bytesnake]
 - Gaussian Mixture Models has been added to `linfa-clustering` by [@relf]

Changes
----------

 - Common metrics for classification and regression have been added
 - A new dataset interface simplifies the work with targets and labels
 - New traits for `Transformer`, `Fit` and `IncrementalFit` standardizes the interface
 - Switched to Github Actions for better integration

Version 0.1.3
===========================

New algorithms 
------------

 - The `DBSCAN` clustering algorithm has been added to `linfa-clustering` ([#12](https://github.com/LukeMathWalker/linfa/pull/12) by [@xd009642])
   
Version 0.1.2 (2019-11-25)
===========================

New algorithms 
------------

 - First release of `linfa-clustering:v0.1.0` with the `KMeans` algorithm (by [@LukeMathWalker])
 - First (real) release of `linfa`, re-exporting `linfa-clustering` (by [@LukeMathWalker])
 

[@LukeMathWalker]: https://github.com/LukeMathWalker
[@xd009642]: https://github.com/xd009642
