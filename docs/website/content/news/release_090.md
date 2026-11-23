+++
title = "Release 0.9.0"
date = "2026-11-23"
+++

Linfa's 0.9.0 release adds Hamerly's accelerated K-means and the Wasserstein distance, brings browser WASM support, and fixes a number of metrics and algorithms whose results did not match their reference definitions.

<!-- more -->

## Breaking changes

Several fixes below change the values returned by existing functions: `explained_variance`, macro F-score, ROC curve/AUC, `Pca::explained_variance`, OPTICS ordering and fitted decision trees. `Pca` models serialized with 0.8.x cannot be deserialized with 0.9.0. The minimum supported Rust version is now 1.88.0.

## Improvements and fixes

* `linfa`:
  * add generic `ResidualChain` composing method by @Feiyang472 in https://github.com/rust-ml/linfa/pull/430
  * add symmetric mean absolute percentage error (sMAPE) by @oglego in https://github.com/rust-ml/linfa/pull/437
  * add browser WASM support (`wasm-bindgen` feature) by @LucaCappelletti94 in https://github.com/rust-ml/linfa/pull/438
  * fix `explained_variance` regression metric to match scikit-learn by @teddytennant in https://github.com/rust-ml/linfa/pull/447
  * fix multiclass macro F-score to average per-class scores by @binggao1230 in https://github.com/rust-ml/linfa/pull/448
  * fix ROC curve to start at the origin when scores are zero by @bardrop in https://github.com/rust-ml/linfa/pull/475
* `linfa-clustering`:
  * fix overflow in Gaussian mixture with log-sum-exp by @schliffen in https://github.com/rust-ml/linfa/pull/444
  * fix OPTICS ordering and reachability of each expansion's start point by @ilgrad in https://github.com/rust-ml/linfa/pull/464
* `linfa-elasticnet`: fix `MultiTaskElasticNet::default_target` shape by @whispem in https://github.com/rust-ml/linfa/pull/462
* `linfa-ensemble`: support serde serialization of ensemble models by @dbrundu in https://github.com/rust-ml/linfa/pull/450
* `linfa-kernel`: vectorise the dense Gaussian kernel by @mtantaoui in https://github.com/rust-ml/linfa/pull/465
* `linfa-lars`: fix the `serde` feature
* `linfa-logistic`:
  * fix label ordering in binary logistic regression by @espenloov in https://github.com/rust-ml/linfa/pull/432
  * add offset support to binary logistic regression by @espenloov in https://github.com/rust-ml/linfa/pull/435
* `linfa-preprocessing`: realign `PreprocessingError` variants with error strings by @oglego in https://github.com/rust-ml/linfa/pull/434
* `linfa-reduction`: fix `Pca::explained_variance` scaling by @AdamEzzat1 in https://github.com/rust-ml/linfa/pull/457
* `linfa-trees`: make `DecisionTree` fitting deterministic and about 3x faster by @mysma-9403 in https://github.com/rust-ml/linfa/pull/452
* `linfa-tsne`: update to `bhtsne` 0.5.4 by @AnthonyMichaelTDM in https://github.com/rust-ml/linfa/pull/429

## New algorithms

### Hamerly's K-means

Hamerly's algorithm accelerates K-means using the triangle inequality and per-sample upper/lower distance bounds, so most distance computations are skipped once the algorithm has stabilised. It gives the same result as Lloyd's algorithm for the same initial centroids. Select it with `KMeans::params(k).algorithm(KMeansAlgorithm::Hamerly)` (batch `fit` only, with a true metric such as `L2Dist`). Added to `linfa-clustering` by @mrsanor in https://github.com/rust-ml/linfa/pull/439

See [Making k-means even faster, G. Hamerly](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf)

### Wasserstein distance

The Wasserstein distance, also known as the Earth Mover's distance, measures the minimal cost of transforming one distribution into another. `WassersteinDist` compares histograms, where each array element is the probability mass at that index. Added to `linfa-nn` by @mrsanor in https://github.com/rust-ml/linfa/pull/440

See [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric)
