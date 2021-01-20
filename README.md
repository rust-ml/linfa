<img align="left" src="./mascot.svg" width="70px" height="70px" alt="Linfa mascot icon">

# Linfa

[![crates.io](https://img.shields.io/crates/v/linfa.svg)](https://crates.io/crates/linfa)
[![Documentation](https://docs.rs/linfa/badge.svg)](https://docs.rs/linfa)
[![Codequality](https://github.com/rust-ml/linfa/workflows/Codequality%20Lints/badge.svg)](https://github.com/rust-ml/linfa/actions?query=workflow%3A%22Codequality+Lints%22)
[![Run Tests](https://github.com/rust-ml/linfa/workflows/Run%20Tests/badge.svg)](https://github.com/rust-ml/linfa/actions?query=workflow%3A%22Run+Tests%22)

> _**linfa**_ (Italian) / _**sap**_ (English):
> 
> The **vital** circulating fluid of a plant.


`linfa` aims to provide a comprehensive toolkit to build Machine Learning applications with Rust.

Kin in spirit to Python's `scikit-learn`, it focuses on common preprocessing tasks and classical ML algorithms for your everyday ML tasks.

_Documentation_: [latest](https://docs.rs/linfa)
_Community chat_: [Zulip](https://rust-ml.zulipchat.com/)

## Current state

Where does `linfa` stand right now? [Are we learning yet?](http://www.arewelearningyet.com/)

`linfa` currently provides sub-packages with the following algorithms: 


| Name | Purpose | Status | Category |  Notes | 
| :--- | :--- | :---| :--- | :---| 
| [clustering](linfa-clustering/) | Data clustering | Tested / Benchmarked  | Unsupervised learning | Clustering of unlabeled data; contains K-Means, Gaussian-Mixture-Model and DBSCAN  | 
| [kernel](linfa-kernel/) | Kernel methods for data transformation  | Tested  | Pre-processing | Maps feature vector into higher-dimensional space| 
| [linear](linfa-linear/) | Linear regression | Tested  | Partial fit | Contains Ordinary Least Squares (OLS), Generalized Linear Models (GLM) | 
| [elasticnet](linfa-elasticnet/) | Elastic Net | Tested | Supervised learning | Linear regression with elastic net constraints |
| [logistic](linfa-logistic/) | Logistic regression | Tested  | Partial fit | Builds two-class logistic regression models
| [reduction](linfa-reduction/) | Dimensionality reduction | Tested  | Pre-processing | Diffusion mapping and Principal Component Analysis (PCA) |
| [trees](linfa-trees/) | Decision trees | Experimental  | Supervised learning | Linear decision trees
| [svm](linfa-svm/) | Support Vector Machines | Tested  | Supervised learning | Classification or regression analysis of labeled datasets | 
| [hierarchical](linfa-hierarchical/) | Agglomerative hierarchical clustering | Tested | Unsupervised learning | Cluster and build hierarchy of clusters |
| [bayes](linfa-bayes/) | Naive Bayes | Tested | Supervised learning | Contains Gaussian Naive Bayes |

We believe that only a significant community effort can nurture, build, and sustain a machine learning ecosystem in Rust - there is no other way forward.

If this strikes a chord with you, please take a look at the [roadmap](https://github.com/rust-ml/linfa/issues/7) and get involved!

## BLAS/Lapack backend

At the moment you can choose between the following BLAS/LAPACK backends: `openblas`, `netblas` or `intel-mkl`

|Backend  | Linux | Windows | macOS |
|:--------|:-----:|:-------:|:-----:|
|OpenBLAS |✔️      |-        |-      |
|Netlib   |✔️      |-        |-      |
|Intel MKL|✔️      |✔️        |✔️      |

For example if you want to use the system IntelMKL library for the PCA example, then pass the corresponding feature:
```
cd linfa-reduction && cargo run --release --example pca --features linfa/intel-mkl-system
```
This selects the `intel-mkl` system library as BLAS/LAPACK backend. On the other hand if you want to compile the library and link it with the generated artifacts, pass `intel-mkl-static`.

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
