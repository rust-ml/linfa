# Kernel methods

`linfa-kernel` provides methods for dimensionality expansion. 

## The Big Picture

`linfa-kernel` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

In machine learning, kernel methods are a class of algorithms for pattern analysis, whose best known member is the [support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine). They owe their name to the kernel functions, which maps the features to some higher-dimensional target space. Common examples for kernel functions are the radial basis function (euclidean distance) or polynomial kernels.

## Current State

linfa-kernel currently provides an implementation of kernel methods for RBF and polynomial kernels, with sparse or dense representation. Further a k-neighbour approximation allows to reduce the kernel matrix size. 

Low-rank kernel approximation are currently missing, but are on the roadmap. Examples for these are the [Nyström approximation](https://www.jmlr.org/papers/volume6/drineas05a/drineas05a.pdf) or [Quasi Random Fourier Features](http://www-personal.umich.edu/~aniketde/processed_md/Stats608_Aniketde.pdf).

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.
