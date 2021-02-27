+++
title = "Release 0.2.0"
date = "2020-11-26"
+++
This release of Linfa introduced 9 new implementations and a couple of changes to the APIs. Travis support for FOSS projects was dropped, so we were forced to switch to Github Actions and we introduced a couple of traits to represent different classes of algorithms in a better way.

<!-- more -->

New algorithms
-----------

 - Ordinary Linear Regression has been added to `linfa-linear` by [@Nimpruda] and [@paulkoerbitz]
 - Generalized Linear Models has been added to `linfa-linear` by [@VasanthakumarV]
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

