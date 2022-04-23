# Elastic Net

`linfa-elasticnet` provides a pure Rust implementations of elastic net linear regression.

## The Big Picture

`linfa-elasticnet` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

The `linfa-elasticnet` crate provides linear regression with ridge and LASSO constraints. The solver uses coordinate descent to find an optimal solution.

This library contains an elastic net implementation for linear regression models. It combines l1 and l2 penalties of the LASSO and ridge methods and offers therefore a greater flexibility for feature selection. With increasing penalization certain parameters become zero, their corresponding variables are dropped from the model.

See also:
 * [Wikipedia on Elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization)

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
$ cargo run --example elasticnet
```

<details>
<summary style="cursor: pointer; display:list-item;">
Show source code
</summary>

```rust
use linfa::prelude::*;
use linfa_elasticnet::{ElasticNet, Result};

// load Diabetes dataset
let (train, valid) = linfa_datasets::diabetes().split_with_ratio(0.90);

// train pure LASSO model with 0.1 penalty
let model = ElasticNet::params()
    .penalty(0.3)
    .l1_ratio(1.0)
    .fit(&train)?;

println!("intercept:  {}", model.intercept());
println!("params: {}", model.hyperplane());

println!("z score: {:?}", model.z_score());

// validate
let y_est = model.predict(&valid);
println!("predicted variance: {}", valid.r2(&y_est)?);
# Result::Ok(())
```
</details>
