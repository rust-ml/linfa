# How to contribute to the Linfa project

This document should be used as a reference when contributing to Linfa. It describes how an algorithm should be implemented to fit well into the Linfa ecosystem. First, there are implementation details, how to use a generic float type, how to use the `Dataset` type in arguments etc. Second, the cargo manifest should be set up, such that a user can choose for different backends. 

## Datasets and learning traits

An important part of the Linfa ecosystem is how to organize data for the training and estimation process. A [Dataset](src/dataset/mod.rs) serves this purpose. It is a small wrapper of data and targets types and should be used as argument for the [Fit](src/traits.rs) trait. Its parametrization is generic, with [Records](src/dataset/mod.rs) representing input data (atm only implemented for `ndarray::ArrayBase`) and [Targets](src/dataset/mod.rs) for targets.

You can find traits for different classes of algorithms [here](src/traits.rs). For example, to implement a fittable algorithm, which takes an `Array2` as input data and boolean array as targets and could fail with an `Error` struct:
```rust
impl<F: Float> Fit<Array2<F>, Array1<bool>, Error> for SvmParams<F, Pr> {
    type Object = Svm<F, Pr>;

    fn fit(&self, dataset: &Dataset<Array2<F>, Array1<bool>>) -> Result<Self::Object, Error> {
        ...
    }
}
```
where the type of the input dataset is `&Dataset<Kernel<F>, Array1<bool>>`. It produces a result with a fitted state, called `Svm<F, Pr>` with probability type `Pr`, or an error of type `Error` in case of failure.

The [Predict](src/traits.rs) trait has its own section later in this document, while for an example of a `Transformer` please look into the [linfa-kernel](linfa-kernel/src/lib.rs) implementation.

## Parameters and builder

An algorithm has a number of hyperparameters, describing how it operates. This section describes how the algorithm's structs should be organized in order to conform with other implementations. 

Imagine we have an implementation of `MyAlg`, there should a separate struct called `MyAlgParams`. The method `MyAlg::params(..) -> MyAlgParams` constructs a parameter set with default parameters and optionally required arguments (for example the number of clusters). If no parameters are required, then `std::default::Default` can be implemented as well:
```rust
impl Default for MyAlgParams {
    fn default() -> MyAlgParams {
        MyAlg::params()
    }
}
```

The `MyAlgParams` should implement the Consuming Builder pattern, explained in the [Rust Book](https://doc.rust-lang.org/1.0.0/style/ownership/builders.html). Each hyperparameter gets a single field in the struct, as well as a method to modify it. Sometimes a random number generator is used in the training process. Then two separate methods should take a seed or a random number generator. With the seed a default RNG is initialized, for example [Isaac64](https://docs.rs/rand_isaac/0.2.0/rand_isaac/isaac64/index.html).

With a constructed set of parameters, the `MyAlgParams::fit(..) -> Result<MyAlg>` executes the learning process and returns a learned state. If one of the parameters is invalid (for example out of a required range), then an `Error::InvalidState` should be returned. For transformers there is only `MyAlg`, and no `MyAlgParams`, because there is no hidden state to be learned.

Following this convention, the pattern can be used by the user like this:
```rust
MyAlg::params()
    .eps(1e-5)
    .backwards(true)
    ...
    .fit(&dataset)?;
```

## Generic float types

Every algorithm should be implemented for `f32` and `f64` floating points. This can be achieved with the `linfa::Float` trait, which is basically just a combination of `ndarray::NdFloat` and `num_traits::Float`. You can look up most of the constants (like zero, one, PI) in the `num_traits` documentation. Here is a small example for a function, generic over `Float`:
```rust
use linfa::Float;
fn div_capped<F: Float>(num: F) {
    F::one() / (num + F::from(1e-5).unwrap())
}
```

## Implement prediction traits

There are two different traits for predictions, `Predict` and `PredictRef`. `PredictRef` takes a reference to the records and produces a new set of targets. This should be implemented by a new algorithms, for example:
```rust
impl<F: Float, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix2>, Array1<F>> for Svm<F, F> {
    fn predict_ref<'a>(&'a self, data: &ArrayBase<D; Ix2>) -> Array1<F> {
        data.outer_iter().map(|data| {
            self.normal.dot(&data) - self.rho
        })
        .collect()
    }
}
```

This implementation is then used by `Predict` to provide the following `records` and `targets` combinations:

 * `Dataset` -> `Dataset`
 * `&Dataset` -> `Array1`
 * `Array2` -> `Dataset`
 * `&Array2` -> `Array1`

and should be imported by the user.

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

## Add a dataset

When you want to add a dataset to the `linfa-datasets` crate, you have to do the following:
 * create a tarball with your dataset as a semicolon separated CSV file and move it to `linfa-datasets/data/?.csv.gz`
 * add a feature with the name of your dataset to `linfa-datasets`
 * create a new function in `linfa-datasets/src/lib.rs` carrying the name of your dataset and loading it as a binary file

For the last step you can look at similar implementations, for example the Iris plant dataset. The idea here is to put the dataset into the produced library directly and parse it from memory. This is obviously only feasible for small datasets.

After adding it to the `linfa-datasets` crate you can include with the corresponding feature to your `Cargo.toml` file
```
linfa-datasets = { version = "0.3.0", path = "../datasets", features = ["winequality"] }
```
and then use it in your example or tests as
```rust
fn main() {
    let (train, valid) = linfa_datasets::winequality()
        .split_with_ratio(0.8);
    /// ...
}
```

## Use the lapack trait bound

When you want to implement an algorithm which requires the [Lapack](https://docs.rs/ndarray-linalg/0.13.1/ndarray_linalg/types/trait.Lapack.html) bound, then you could add the trait bound to the `linfa::Float` standard bound, e.g. `F: Float + Scalar + Lapack`. If you do that you're currently running into conflicting function definitions of [num_traits::Float](https://docs.rs/num-traits/0.2.14/num_traits/float/trait.Float.html) and [cauchy::Scalar](https://docs.rs/cauchy/0.4.0/cauchy/trait.Scalar.html) with the first defined for real-valued values and the second for complex values. 

If you want to avoid that you can use the `linfa::dataset::{WithLapack, WithoutLapack}` traits, which basically adds the lapack trait bound for a block and then removes it again so that the conflicts can be avoided. For example:
```rust
let decomp = covariance.with_lapack().cholesky(UPLO::Lower)?;
let sol = decomp
     .solve_triangular(UPLO::Lower, Diag::NonUnit, &Array::eye(n_features))?
     .without_lapack();
```
