+++
title = "About"
date = 2025-12-23
+++

Linfa aims to provide a comprehensive toolkit to build Machine Learning applications with Rust.

Kin in spirit to Python's `scikit-learn`, it focuses on common preprocessing tasks and classical ML algorithms for your everyday ML tasks.

## Current state

Where does `linfa` stand right now? [Are we learning yet?](http://www.arewelearningyet.com/)

`linfa` currently provides sub-packages with the following algorithms: 


<div class="outer-table">

| Name                                                                                          | Purpose                                  | Status                | Category              | Notes                                                                                     |
| :-------------------------------------------------------------------------------------------- | :--------------------------------------- | :-------------------- | :-------------------- | :---------------------------------------------------------------------------------------- |
| [bayes](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-bayes/)                 | Naive Bayes                              | Tested                | Supervised learning   | Contains Bernouilli, Gaussian and Multinomial Naive Bayes                                 |
| [clustering](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-clustering/)       | Data clustering                          | Tested / Benchmarked  | Unsupervised learning | Clustering of unlabeled data; contains K-Means, Gaussian-Mixture-Model, DBSCAN and OPTICS |
| [ensemble](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-ensemble/)           | Ensemble methods                         | Tested                | Supervised learning   | Contains bagging, random forest and AdaBoost                                              |
| [elasticnet](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-elasticnet/)       | Elastic Net                              | Tested                | Supervised learning   | Linear regression with elastic net constraints                                            |
| [ftrl](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-ftrl/)                   | Follow The Regularized Leader - proximal | Tested  / Benchmarked | Partial fit           | Contains L1 and L2 regularization. Possible incremental update                            |
| [hierarchical](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-hierarchical/)   | Agglomerative hierarchical clustering    | Tested                | Unsupervised learning | Cluster and build hierarchy of clusters                                                   |
| [ica](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-ica/)                     | Independent component analysis           | Tested                | Unsupervised learning | Contains FastICA implementation                                                           |
| [kernel](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-kernel/)               | Kernel methods for data transformation   | Tested                | Pre-processing        | Maps feature vector into higher-dimensional space                                         |
| [lars](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-lars/)                   | Linear regression                        | Tested                | Supervised learning   | Contains Least Angle Regression (LARS)                                                    |
| [linear](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-linear/)               | Linear regression                        | Tested                | Supervised learning   | Contains Ordinary Least Squares (OLS), Generalized Linear Models (GLM)                    |
| [logistic](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-logistic/)           | Logistic regression                      | Tested                | Partial fit           | Builds two-class logistic regression models                                               |
| [nn](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-nn/)                       | Nearest Neighbours & Distances           | Tested / Benchmarked  | Pre-processing        | Spatial index structures and distance functions                                           |
| [pls](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-pls/)                     | Partial Least Squares                    | Tested                | Supervised learning   | Contains PLS estimators for dimensionality reduction and regression                       |
| [preprocessing](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-preprocessing/) | Normalization & Vectorization            | Tested / Benchmarked  | Pre-processing        | Contains data normalization/whitening and count vectorization/tf-idf                      |
| [reduction](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-reduction/)         | Dimensionality reduction                 | Tested                | Pre-processing        | Diffusion mapping, Principal Component Analysis (PCA), Random projections                 |
| [svm](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-svm/)                     | Support Vector Machines                  | Tested                | Supervised learning   | Classification or regression analysis of labeled datasets                                 |
| [trees](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-trees/)                 | Decision trees                           | Tested / Benchmarked  | Supervised learning   | Linear decision trees                                                                     |
| [tsne](https://github.com/rust-ml/linfa/tree/master/algorithms/linfa-tsne/)                   | Dimensionality reduction                 | Tested                | Unsupervised learning | Contains exact solution and Barnes-Hut approximation t-SNE                                |


</div>

We believe that only a significant community effort can nurture, build, and sustain a machine learning ecosystem in Rust - there is no other way forward.

If this strikes a chord with you, please take a look at the [roadmap](https://github.com/rust-ml/linfa/issues/7) and get involved!
