# linfa

[![Build Status](https://travis-ci.org/rust-ml/linfa.svg?branch=master)](https://travis-ci.org/rust-ml/linfa)

> _**linfa**_ (Italian) / _**sap**_ (English):
> 
> The **vital** circulating fluid of a plant.


`linfa` aims to provide a comprehensive toolkit to build Machine Learning applications with Rust.

Kin in spirit to Python's `scikit-learn`, it focuses on common preprocessing tasks and classical ML algorithms for your everyday ML tasks.

_Documentation_: [latest](https://docs.rs/linfa)
_Community chat_: [Zulip](https://rust-ml.zulipchat.com/)

## Current state

Such bold ambitions! Where are we now? [Are we learning yet?](http://www.arewelearningyet.com/)

`linfa` currently provides sub-packages with the following algorithms: 
- `clustering`: Clustering of unlabeled data
    - K-Means and DBSCAN
- `kernel`    
- `linear`: Linear regression 
    - Ordinary Least Squares (OLS)
- `logistic`: Logistic Regression
    - Two-class logsitic regression models
- `reduction`: Dimensional reduction
    - Diffusion mapping
    - Principal Component Analysis (PCA)
- `svm`: Support Vector Machines
- `trees`: Decision trees

We believe that only a significant community effort can nurture, build, and sustain a machine learning ecosystem in Rust - there is no other way forward.

If this strikes a chord with you, please take a look at the [roadmap](https://github.com/rust-ml/linfa/issues) and get involved!

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
