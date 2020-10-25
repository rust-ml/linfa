# How to contribute to the Linfa project

This document should be used as a reference when contributing to Linfa. It describes how an algorithm should be implemented to fit well into Linfas ecosystem. First, there are implementation details, how to use a generic float type, how to accept a `Dataset` etc. Second, the cargo manifest should be set up, such that a user can choose for different backends. 

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

Every algorithm should be implemented for `f32` and `f64` floating points. This can be achieved with the `linfa::Float` trait, which is basically just a combination of `ndarray::NdFloat` and `num_traits::Float`. You can look up most of the constants (like zero, one, PI) in the `num_traits` documentation. Here is a small example for a function, generic over `Float`:
```rust
use linfa::Float;
fn div_capped<F: Float>(num: F) {
    F::one() / (num + F::from(1e-5).unwrap())
}
```

## Datasets and learning traits

An important part of the Linfa ecosystem is how to organize data for the training and estimation process. A [Dataset](src/dataset/mod.rs) serves this purpose. It is a small wrapper of data and targets types and should be used as argument for the [Fit](src/traits.rs) trait. Its parametrization is generic, with [Records](src/dataset/mod.rs) representing input data (atm only implemented for `ndarray::ArrayBase`) and [Targets](src/dataset/mod.rs) for targets.

You can find traits for different classes of algorithms [here](src/traits.rs). For example, to implement a fittable algorithm, which takes a `Kernel` as input data and boolean array as targets:
```rust
impl<'a, F: Float> Fit<'a, Kernel<'a, F>, Vec<bool>> for SvmParams<F, Pr> {
    type Object = Svm<'a, F, Pr>;

    fn fit(&self, dataset: &'a Dataset<Kernel<'a, F>, Vec<bool>>) -> Self::Object {
        ...
    }
}
```
the type of the dataset is `&'a Dataset<Kernel<'a, F>, Vec<bool>>`, ensuring that the kernel lives long enough for the fitting process. It produces a fitted state, called `Svm<'a, F, Pr>` with probability type `Pr`.

The [Predict](src/traits.rs) should be implemented for datasets, as well as arrays. If a dataset is provided, then predict takes its ownership and returns a new dataset with overwritten targets. For an array, predict takes a reference and returns predicted targets. In the same context, SVM implemented predict like this:
```rust
impl<'a, F: Float, T: Targets> Predict<Dataset<Array2<F>, T>, Dataset<Array2<F>, Vec<Pr>>>
    for Svm<'a, F, Pr>
{
    fn predict(&self, data: Dataset<Array2<F>, T>) -> Dataset<Array2<F>, Vec<Pr>> {
        ...
    }
}
```
and
```rust
impl<'a, F: Float, D: Data<Elem = F>> Predict<ArrayBase<D, Ix2>, Vec<Pr>> for Svm<'a, F, Pr> {
    fn predict(&self, data: ArrayBase<D, Ix2>) -> Vec<Pr> {
        ...
    }
}
```

For an example of a `Transformer` please look into the [linfa-kernel](linfa-kernel/src/lib.rs) implementation.

## Make serde optionally

If you want to implement `Serialize` and `Deserialize` for your parameters, please do that behind a feature flag. You can add to your cargo manifest
```
[features]
serde = ["serde_crate", "ndarray/serde"]

[dependencies.serde_crate]
package = "serde"
optional = true
version = "1.0"
```
which basically renames the `serde` crate to `serde_crate` and adds a feature `serde`. In your parameter struct, move the macro definition behind the `serde` feature:
```rust
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
pub struct HyperParams {
...
}
```

