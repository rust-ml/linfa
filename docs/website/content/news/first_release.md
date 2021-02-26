+++
title = "Release 0.2.0"
date = "2020-02-14"
+++
Linfa 0.2.0 provides 9 new algorithms and improves working with them.
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

