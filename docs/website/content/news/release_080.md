+++
title = "Release 0.8.0"
date = "2025-09-30"
+++

Besides Bernouilli naive bayes classifier and bootstrap aggregation algorithm, most notably Linfa's 0.8.0 release brings support for `ndarray` 0.16.

<!-- more -->

## Improvements and fixes

 * add `max_features` and `tokenizer_function` to `CountVectorizer` in `linfa-preprocessing`
 * add `predict_proba()` to `Gaussian mixture model` in `linfa-clustering` 
 * add `predict_proba()` and `predict_log_proba()` to algorithms in `linfa-bayes`
 * add target names to `dataset`
 * fix SVR parameterization in `linfa-svm`
 * fix serde support for algorithms in `linfa-pls`
 * fix confusion matrix: use predicted and ground thruth labels, make it reproducible
 * fix dataset names after shuffling 
 * bump `ndarray` to 0.16, `argmin` to 0.11.0, `kdtree` to 0.7.0, statrs to `0.18`, sprs to `0.11`
 * bump MSRV to 1.87.0

## New algorithms

### Bernouilli Naive Bayes

Naive Bayes for Bernouilli models is a classification algorithm for data that is distributed according to multivariate Bernoulli distributions; 
i.e., there may be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean) variable. 

See [scikit-learn.naive_bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

### Bootstrap aggregation 

In ensemble algorithms, bagging (Bootstrap aggregating) methods form a class of algorithms which build several instances of a black-box 
estimator on random subsets of the original training set and then aggregate their individual predictions to form a final prediction.

See [sklearn.ensemble](https://scikit-learn.org/stable/modules/ensemble.html)
