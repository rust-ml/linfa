# Datasets

`linfa-datasets` provides a collection of commonly used datasets ready to be used in tests and examples.

## The Big Picture

`linfa-datasets` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current State

Currently the following datasets are provided:

* `["iris"]` : iris flower dataset
* `["winequality"]` : wine quality dataset
* `["diabetes"]` : diabetes dataset

along with methods to easily load them. Loaded datasets are returned as a [`linfa::Dataset`](https://docs.rs/linfa/0.3.0/linfa/dataset/type.Dataset.html) structure whith named features. 

## Using a dataset

To use one of the provided datasets in your project add the crate to your Cargo.toml with the corresponding feature enabled:
```
linfa-datasets = { version = "0.3.0", features = ["winequality"] }
```
and then use it in your example or tests as
```rust 
fn main() {
    let (train, valid) = linfa_datasets::winequality()
        .split_with_ratio(0.8);
    /// ...
}
```
