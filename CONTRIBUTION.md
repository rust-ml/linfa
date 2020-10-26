# How to contribute to the Linfa project

This document should be used as a reference when contributing to Linfa. It describes how an algorithm should be implemented to fit well into the general ecosystem. First, there are implementation details, how to use a generic float type, how to accept a `Dataset` etc. Second, the cargo manifest should be set up, such that a user can choose for different backends. 

## Let the user choose their favourite linear algebra library

One important decision for the user of Linfa is the linear algebra library backend. The `ndarray-linalg` library supports at the moment `openblas`, `netblas` or `intel MKL` as backends. It is considered good practice to allow the user to choose, which backend they wants to use. Let's say you're using the `linfa-kernel` subcrate and have added `ndarray-linalg` as a dependency as well, then your cargo manifest should add the corresponding features:

```
[features]
default = []
openblas = ["ndarray-linalg/openblas", "linfa-kernel/openblas"]
intel-mkl = ["ndarray-linalg/intel-mkl", "linfa-kernel/intel-mkl"]
netlib = ["ndarray-linalg/netlib", "linfa-kernel/netlib"]

[dependencies]
ndarray = { version = "0.13", default-features=false, features=["blas"] }
ndarray-linalg = { version = "0.12" }
linfa-kernel = { path = "../linfa-kernel" }
...

```

## Use a specific backend for testing

When you're implementing tests, relying on `ndarray-linalg`, you have to add the `openblas-src` crate. This will instruct cargo to compile the backend, in order to find the required symbols. Your cargo manifest should have a dependency
```
[dev-dependencies]
...
openblas-src = "0.9" 
```
and you have to add an `extern crate openblas_src` in the `tests` module.

## Generic float types

As always, it's nice to have a separate implementation for `f32` and `f64` datatypes. This can be achieved with the `linfa::Float` trait, which is basically just a combination of `ndarray::NdFloat` and `num_traits::Float`. You can look up most of the constants (like zero, one, PI) in the `num_traits` documentation. Here is a small example for a function, generic over `Float`:
```rust
use linfa::Float;
fn div_capped<F: Float>(num: F) {
    F::one() / (num + F::from(1e-5).unwrap())
}
```

## Use datasets
