# Support Vector Machines

`linfa-svm` provides a pure Rust implementation for support  vector machines. 

## The Big Picture

`linfa-svm` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

Support Vector Machines are one major branch of machine learning models and offer classification or regression analysis of labeled datasets. They seek a discriminant, which seperates the data in an optimal way, e.g. have the fewest numbers of miss-classifications and maximizes the margin between positive and negative classes. A support vector contributes to the discriminant and is therefore important for the classification/regression task. The balance between the number of support vectors and model performance can be controlled with hyperparameters.

## Current State

 linfa-svm currently provides an implementation of SVM with Sequential Minimal Optimization:
  - Support Vector Classification with C/Nu/one-class
  - Support Vector Regression with Epsilon/Nu


## Examples

There is an usage example in the `examples/` directory. To run, use:

```bash
$ cargo run --release --example winequality
```

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.