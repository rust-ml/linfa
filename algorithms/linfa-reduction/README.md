# Dimensional Reduction

`linfa-reduction` aims to provide pure Rust implementations of dimensional reduction algorithms. 

## The Big Picture

`linfa-reduction` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-reduction` currently provides an implementation of the following dimensional reduction methods: 
- Diffusion Mapping
- Principal Component Analysis (PCA)

## BLAS/Lapack backend

By default, we use a pure-Rust implementation for all linear algebra routines. However, you can also choose an external BLAS/LAPACK backend library instead, by enabling the `blas` feature and a feature corresponding to your BLAS backend. Currently you can choose between the following BLAS/LAPACK backends: `openblas`, `netblas` or `intel-mkl`.

|Backend  | Linux | Windows | macOS |
|:--------|:-----:|:-------:|:-----:|
|OpenBLAS |✔️      |-        |-      |
|Netlib   |✔️      |-        |-      |
|Intel MKL|✔️      |✔️        |✔️      |

Each BLAS backend has two features available. The feature allows you to choose between linking the BLAS library in your system or statically building the library. For example, the features for the `intel-mkl` backend are `intel-mkl-static` and `intel-mkl-system`.

An example set of Cargo flags for enabling the Intel MKL backend is `--features blas,linfa/intel-mkl-system`. Note that the backend features are defined on the `linfa` crate.

## Examples

There is an usage example in the `examples/` directory. To run, use:

```bash
$ cargo run --release --example diffusion_map
$ cargo run --release --example pca
```
