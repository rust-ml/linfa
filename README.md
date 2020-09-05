# linfa
[![crates.io](https://img.shields.io/crates/v/linfa.svg)](https://crates.io/crates/linfa)
[![Documentation](https://docs.rs/linfa/badge.svg)](https://docs.rs/linfa)
[![Build Status](https://travis-ci.org/rust-ml/linfa.svg?branch=master)](https://travis-ci.org/rust-ml/linfa)

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
| clustering | Data clustering | Tested  | Unsupervised learning | Clustering of unlabled data; contains K-Means and DBSCAN  | 
| kernel | Kernel methods for data transformation  | Tested  | Pre-processing | Maps feature vector into higher-dimensional space| 
| linear | Linear regression | Tested  | Partial fit | Contains Ordinary Least Squares (OLS) | 
| logistic | Logistic regression | Tested  | Partial fit | Builds two-class logistic regression models
| reduction | Dimensional reduction | Tested  | Pre-processing | Diffusion mapping and Principal Component Analysis (PCA) |
| trees | Decision trees | Tested  | Supervised learning | Linear decision trees
| svm | Support Vector Machines | Tested  | Supervised learning | Classification or regression analysis of labeled datasets | 

We believe that only a significant community effort can nurture, build, and sustain a machine learning ecosystem in Rust - there is no other way forward.

If this strikes a chord with you, please take a look at the [roadmap](https://github.com/rust-ml/linfa/issues) and get involved!

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
