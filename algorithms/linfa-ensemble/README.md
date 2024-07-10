# Enseble Learning

`linfa-ensemble` provides pure Rust implementations of Ensemble Learning algorithms for the Linfa toolkit.

## The Big Picture

`linfa-ensemble` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-ensemble` currently provides an implementation of bootstrap aggregation (bagging) for other classifers provided in linfa.

## Examples

You can find examples in the `examples/` directory. To run an bootstrap aggregation for ensemble of decision trees (a Random Forest) use:

```bash
$ cargo run --example randomforest_iris --release
```


