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

## Parameters and checking

An algorithm has a number of hyperparameters, describing how it operates. This section describes how the algorithm's structs should be organized in order to conform with other implementations. 

Sometimes only an algorithm's parameters must be checked for validity. As such, Linfa makes a distinction between checked and unchecked parameters. Unchecked parameters can be converted into checked parameters if the values are valid, and only checked parameters can be used to run the algorithm.

Imagine we have an implementation of `MyAlg`, there should separate structs called `MyAlgValidParams`, which are the checked parameters, and `MyAlgParams`, which are the unchecked parameters. The method `MyAlg::params(..) -> MyAlgParams` constructs a parameter set with default parameters and optionally required arguments (for example the number of clusters). `MyAlgValidParams` should be a struct that contains all the hyperparameters as fields, and `MyAlgParams` should just be a newtype that wraps `MyAlgValidParams`.
```rust
struct MyAlgValidParams {
    eps: f32,
    backwards: bool,
}

struct MyAlgParams(MyAlgValidParams);
```

`MyAlgParams` should implement the Consuming Builder pattern, explained in the [Rust Book](https://doc.rust-lang.org/1.0.0/style/ownership/builders.html). Each hyperparameter gets a method to modify it. `MyAlgParams` should also implement the `ParamGuard` trait, which facilitates parameter checking. The associated type `ParamGuard::Checked` should be `MyAlgValidParams` and the `check_ref()` method should contain the parameter checking logic, while `check()` simply calls `check_ref()` before unwrapping the inner `MyAlgValidParams`.

With a checked set of parameters, `MyAlgValidParams::fit(..) -> Result<MyAlg>` executes the learning process and returns a learned state. Due to blanket impls on `ParamGuard`, it's also possible to call `fit()` or `transform()` directly on `MyAlgParams` as well, which performs the parameter checking before the learning process.

Following this convention, the pattern can be used by the user like this:
```rust
MyAlg::params()
    .eps(1e-5)
    .backwards(true)
    ...
    .fit(&dataset)?;
```
or, if the checking is done explicitly:
```rust
let params = MyAlg::params().check();
params.fit(&dataset);
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

There are two different traits for predictions, `Predict` and `PredictInplace`. `PredictInplace` takes a reference to the records and writes the prediction into a target parameter. This should be implemented by a new algorithms, for example:
```rust
impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<F>> for Svm<F, F> {
    fn predict_inplace(&self, data: &ArrayBase<D, Ix2>, targets: &mut Array1<F>) {
        assert_eq!(data.n_rows(), targets.len(), "The number of data points must match the number of output targets.");

        for (data, target) in data.outer_iter().zip(targets.iter_mut()) {
            *target = self.normal.dot(&data) - self.rho;
        }
    }

    fn default_target(&self, data: &ArrayBase<D, Ix2>) -> Array1<F> {
        Array1::zeros(data.nrows())
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

## Benchmarking

### Building Benchmarks
It is important to the project that we have benchmarks in place to evaluate the benefit of performance related changes. To make that process easier we provide some guidelines for writing benchmarks.

1. Test for a variety of sample sizes for most algorithms [1_000, 10_000, 20_000] will be sufficient. For algorithms where it's not too slow, use 100k instead of 20k.
2. Test for a variety of feature dimensions. Two is usually enough for most algorithms. The following are suggested feature dimensions to use for different types of algorithms:
    - Spatial algorithms: [3, 8]
    - Dimentionality reduction algorithms: [10, 20, 50]
    - Others: [5, 10]
3. Use Criterion. Iai another popular benchmarking tool is not being actively maintained at the moment and at the time of this writing it hasn't been updated since Feb 25, 2021.
4. Test various alg implementations for instance Pls has the following algorithms: Nipals and Svd.
5. For algorithms that require an RNG or random seed as input, use a constant seed for reproducibility
6. When benchmarking multi-target the target count should be within the following range: [2, 4].
7. In `BenchmarkId` include the values used to parametrize the benchmark. For example if we're doing Pls then we may have something like `Canonical-Nipals-5feats-1_000samples`
8. Pass data as an argument to the function being benched. This will prevent Criterion from including data creation time as part of the benchmark.
9. Add a profiler see [here](https://github.com/tikv/pprof-rs#integrate-with-criterion) for an example on how to do so with pprof, Criterion, and Flamegraph.

Feel free to use the pls bench as a guideline. Note that it uses functions get get default configurations for profiling and benchmarking. In most cases you can copy and paste such those portions of code. If other configurations are desired it is still easily customizable and explained in the pprof and Criterion documentation.

### Running Benchmarks
When running benchmarks sometimes you will want to profile the code execution. Assuming you have followed step 9 to add a pprof profiling hook for the linfa-ica package you can run the following to get your profiling results as a flamegraph.

`cargo bench -p linfa-ica --bench fast_ica -q -- --profile-time 30`

If you are interested in running a regular criterion bench for linfa-ica then you can run the following

`cargo bench -p linfa-ica`

### Reporting Benchmark Metrics
It is important that we have a consistent methodology for reporting benchmarks below is a template that should aid reviewers.

```
### Context
In a bullet list describe the following:
1. Run on battery charge or while plugged in
2. Power saving mode
3. If the computer was idle during benchmark
4. If the computer was overheating
5. Hardware specs

### Bench Command Run
bench results (code format)

[Attached Flamegraphs if profile runs were also done]
```