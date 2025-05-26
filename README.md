<img align="left" src="./mascot.svg" width="70px" height="70px" alt="Linfa mascot icon">

# Linfa

[![crates.io](https://img.shields.io/crates/v/linfa.svg)](https://crates.io/crates/linfa)
[![Documentation](https://docs.rs/linfa/badge.svg)](https://docs.rs/linfa)
[![DocumentationLatest](https://img.shields.io/badge/docs-latest-blue)](https://rust-ml.github.io/linfa/rustdocs/linfa/)
[![Codequality](https://github.com/rust-ml/linfa/workflows/Codequality%20Lints/badge.svg)](https://github.com/rust-ml/linfa/actions?query=workflow%3A%22Codequality+Lints%22)
[![Run Tests](https://github.com/rust-ml/linfa/workflows/Run%20Tests/badge.svg)](https://github.com/rust-ml/linfa/actions?query=workflow%3A%22Run+Tests%22)

> _**linfa**_ (Italian) / _**sap**_ (English):
> 
> The **vital** circulating fluid of a plant.


`linfa` aims to provide a comprehensive toolkit to build Machine Learning applications with Rust.

Kin in spirit to Python's `scikit-learn`, it focuses on common preprocessing tasks and classical ML algorithms for your everyday ML tasks.

<strong>
    <a href="https://rust-ml.github.io/linfa/">Website</a> | <a href="https://rust-ml.zulipchat.com">Community chat</a>
</strong>

## Current state

Where does `linfa` stand right now? [Are we learning yet?](http://www.arewelearningyet.com/)

`linfa` currently provides sub-packages with the following algorithms: 


| Name | Purpose | Status | Category |  Notes | 
| :--- | :--- | :---| :--- | :---| 
| [bayes](algorithms/linfa-bayes/) | Naive Bayes | Tested | Supervised learning | Contains Bernouilli, Gaussian and Multinomial Naive Bayes |
| [clustering](algorithms/linfa-clustering/) | Data clustering | Tested / Benchmarked  | Unsupervised learning | Clustering of unlabeled data; contains K-Means, Gaussian-Mixture-Model, DBSCAN and OPTICS | 
| [ensemble](algorithms/linfa-ensemble/) | Ensemble methods | Tested | Supervised learning | Contains bagging |
| [elasticnet](algorithms/linfa-elasticnet/) | Elastic Net | Tested | Supervised learning | Linear regression with elastic net constraints |
| [ftrl](algorithms/linfa-ftrl/) | Follow The Regularized Leader - proximal | Tested  / Benchmarked | Partial fit | Contains L1 and L2 regularization. Possible incremental 
| [hierarchical](algorithms/linfa-hierarchical/) | Agglomerative hierarchical clustering | Tested | Unsupervised learning | Cluster and build hierarchy of clusters |
| [ica](algorithms/linfa-ica/) | Independent component analysis | Tested | Unsupervised learning | Contains FastICA implementation |
| [kernel](algorithms/linfa-kernel/) | Kernel methods for data transformation  | Tested  | Pre-processing | Maps feature vector into higher-dimensional space| 
| [linear](algorithms/linfa-linear/) | Linear regression | Tested  | Partial fit | Contains Ordinary Least Squares (OLS), Generalized Linear Models (GLM) | 
| [logistic](algorithms/linfa-logistic/) | Logistic regression | Tested  | Partial fit | Builds two-class logistic regression models
| [nn](algorithms/linfa-nn/) | Nearest Neighbours & Distances | Tested / Benchmarked | Pre-processing | Spatial index structures and distance functions |
| [pls](algorithms/linfa-pls/) | Partial Least Squares | Tested | Supervised learning | Contains PLS estimators for dimensionality reduction and regression |
| [preprocessing](algorithms/linfa-preprocessing/) |Normalization & Vectorization| Tested / Benchmarked | Pre-processing | Contains data normalization/whitening and count 
| [reduction](algorithms/linfa-reduction/) | Dimensionality reduction | Tested | Pre-processing | Diffusion mapping, Principal Component Analysis (PCA), Random projections |
| [svm](algorithms/linfa-svm/) | Support Vector Machines | Tested  | Supervised learning | Classification or regression analysis of labeled datasets | 
| [trees](algorithms/linfa-trees/) | Decision trees | Tested / Benchmarked  | Supervised learning | Linear decision trees
| [tsne](algorithms/linfa-tsne/) | Dimensionality reduction| Tested | Unsupervised learning | Contains exact solution and Barnes-Hut approximation t-SNE |
vectorization/tf-idf |
update |

We believe that only a significant community effort can nurture, build, and sustain a machine learning ecosystem in Rust - there is no other way forward.

If this strikes a chord with you, please take a look at the [roadmap](https://github.com/rust-ml/linfa/issues/7) and get involved!

## BLAS/Lapack backend

Some algorithm crates need to use an external library for linear algebra routines. By default, we use a pure-Rust implementation. However, you can also choose an external BLAS/LAPACK backend library instead, by enabling the `blas` feature and a feature corresponding to your BLAS backend. Currently you can choose between the following BLAS/LAPACK backends: `openblas`, `netblas` or `intel-mkl`.

|Backend  | Linux | Windows | macOS |
|:--------|:-----:|:-------:|:-----:|
|OpenBLAS |✔️      |-        |-      |
|Netlib   |✔️      |-        |-      |
|Intel MKL|✔️      |✔️        |✔️      |

Each BLAS backend has two features available. The feature allows you to choose between linking the BLAS library in your system or statically building the library. For example, the features for the `intel-mkl` backend are `intel-mkl-static` and `intel-mkl-system`.

An example set of Cargo flags for enabling the Intel MKL backend on an algorithm crate is `--features blas,linfa/intel-mkl-system`. Note that the BLAS backend features are defined on the `linfa` crate, and should only be specified for the final executable.

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
