+++
title = "About"
date = 2021-02-26
+++

Linfa aims to provide a comprehensive toolkit to build Machine Learning applications with Rust.

Kin in spirit to Python's `scikit-learn`, it focuses on common preprocessing tasks and classical ML algorithms for your everyday ML tasks.

## Current state

Where does `linfa` stand right now? [Are we learning yet?](http://www.arewelearningyet.com/)

`linfa` currently provides sub-packages with the following algorithms: 


<div class="outer-table">

| Name | Purpose | Status | Category |  Notes | 
| :--- | :--- | :---| :--- | :---| 
| [clustering](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-clustering) | Data clustering | Tested / Benchmarked  | Unsupervised learning | Clustering of unlabeled data; contains K-Means, Gaussian-Mixture-Model and DBSCAN  | 
| [kernel](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-kernel) | Kernel methods for data transformation  | Tested  | Pre-processing | Maps feature vector into higher-dimensional space| 
| [linear](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-linear) | Linear regression | Tested  | Partial fit | Contains Ordinary Least Squares (OLS), Generalized Linear Models (GLM) | 
| [elasticnet](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-elasticnet) | Elastic Net | Tested | Supervised learning | Linear regression with elastic net constraints |
| [logistic](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-logistic) | Logistic regression | Tested  | Partial fit | Builds two-class logistic regression models
| [reduction](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-reduction) | Dimensionality reduction | Tested  | Pre-processing | Diffusion mapping and Principal Component Analysis (PCA) |
| [trees](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-trees) | Decision trees | Experimental  | Supervised learning | Linear decision trees
| [svm](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-svm) | Support Vector Machines | Tested  | Supervised learning | Classification or regression analysis of labeled datasets | 
| [hierarchical](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-hierarchical) | Agglomerative hierarchical clustering | Tested | Unsupervised learning | Cluster and build hierarchy of clusters |
| [bayes](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-bayes) | Naive Bayes | Tested | Supervised learning | Contains Gaussian Naive Bayes |
| [ica](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-ica) | Independent component analysis | Tested | Unsupervised learning | Contains FastICA implementation |

</div>

We believe that only a significant community effort can nurture, build, and sustain a machine learning ecosystem in Rust - there is no other way forward.

If this strikes a chord with you, please take a look at the [roadmap](https://github.com/rust-ml/linfa/issues/7) and get involved!
