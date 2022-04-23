# Clustering

`linfa-clustering` aims to provide pure Rust implementations of popular clustering algorithms.

## The big picture

`linfa-clustering` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

You can find a roadmap (and a selection of good first issues)
[here](https://github.com/rust-ml/linfa/issues) - contributors are more than welcome!

## Current state

`linfa-clustering` currently provides implementation of the following clustering algorithms, in addition to a couple of helper functions: 
- K-Means
- DBSCAN
- Approximated DBSCAN
- Gaussian Mixture Model


Implementation choices, algorithmic details and a tutorial can be found 
[here](https://docs.rs/linfa-clustering).

**WARNING:** Currently the Approximated DBSCAN implementation is slower than the normal DBSCAN implementation. Therefore DBSCAN should always be used over Approximated DBSCAN.

## BLAS/Lapack backend

By default, we use a pure-Rust implementation for all linear algebra routines. However, you can also choose an external BLAS/LAPACK backend library instead, by enabling the `blas` feature and a feature corresponding to your BLAS backend. Currently you can choose between the following BLAS/LAPACK backends: `openblas`, `netblas` or `intel-mkl`.

|Backend  | Linux | Windows | macOS |
|:--------|:-----:|:-------:|:-----:|
|OpenBLAS |✔️      |-        |-      |
|Netlib   |✔️      |-        |-      |
|Intel MKL|✔️      |✔️        |✔️      |

Each BLAS backend has two features available. The feature allows you to choose between linking the BLAS library in your system or statically building the library. For example, the features for the `intel-mkl` backend are `intel-mkl-static` and `intel-mkl-system`.

An example set of Cargo flags for enabling the Intel MKL backend is `--features blas,linfa/intel-mkl-system`. Note that the backend features are defined on the `linfa` crate.

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
