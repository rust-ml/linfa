+++
title = "Release 0.4.0"
date = "2021-04-27"
+++

Linfa's 0.4.0 release introduced four new algorithms, improved documentation of ICA and K-means, added more benchmarks to K-Means and updated to ndarray's 0.14 version.

<!-- more -->

## New algorithms

The [Partial Least Squares Regression](https://en.wikipedia.org/wiki/Partial_least_squares_regression) model family is added in this release. It projects the observable, as well as predicted variables to a latent space and maximizes the correlation in it. For problems with a large number of targets or collinear predictors it gives a better performance when compared to standard regression. For more information look into the documentation of `linfa-psa`.

A wrapper for Barnes-Hut t-SNE is also added in this release. The t-SNE algorithm is often used for data visualization and projects data in a high-dimensional space to a similar representation in two/three dimension. It does so by maximizing the Kullback-Leibler Divergence between the high dimensional source distribution to the target distribution. The Barnes-Hut approximation improves the runtime drastically while retaining the performance. Kudos to [github/frjnn](https://github.com/frjnn/) for providing an implementation!

A new preprocessing crate makes working with textual data and data normalization easy. It implements _count-vectorizer_ and _IT-IDF_ normalization for text pre-processing. Normalizations for signals include linear scaling, norm scaling and whitening with PCA/ZCA/choelsky. An example with a Naive Bayes model achieves 84% F1 score for predicting categories `alt.atheism`, `talk.religion.misc`, `comp.graphics` and `sci.space` on a news dataset.

[Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) calibrates a real-valued classification model to probabilities over two classes. This is used for the SV classification when probabilities are predicted. Further a multi class model, combining multiple binary models (e.g. calibrated SVM models) into a single multi-class model is also added. These composing models are moved to the `linfa/src/composing/` subfolder.

## Improvements

Numerous improvements are added to the KMeans implementation, thanks to @YuhanLiin. The implementation is optimized for single-shot training, an incremental training model is added and KMeans++/KMeans|| initialization gives good initial cluster means for medium and large datasets.

We also moved to ndarray's version 0.14 and introduced `F::cast` for simpler floating point casting.

Other changes:

 * fix for border points in the DBSCAN implementation
 * improved documentation of the ICA subcrate
 * prevent overflowing code example in website

# Example

