# Preprocessing
## The Big Picture

`linfa-preprocessing` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state
`linfa-preprocessing` provides a pure Rust implementation of:
* Standard scaling 
* Min-max scaling
* Max Abs Scaling
* Normalization
* Count vectorization
* TfIdf vectorization
* Whitening

## Examples

There are various usage examples in the `examples/` directory. To run, use:

```bash
$ cargo run --release --example count_vectorization
```
```bash
$ cargo run --release --example tfidf_vectorization
```
```bash
$ cargo run --release --example scaling
```
```bash
$ cargo run --release --example whitening
```

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

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.
