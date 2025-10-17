# Ensemble Learning

`linfa-ensemble` provides pure Rust implementations of Ensemble Learning algorithms for the Linfa toolkit.

## The Big Picture

`linfa-ensemble` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-ensemble` currently provides an implementation of bootstrap aggregation (bagging) for other classifiers provided in linfa.

## Examples

You can find examples in the `examples/` directory. To run an bootstrap aggregation for ensemble of decision trees (a Random Forest) use:

```bash
$ cargo run --example ensemble_iris --release
```

The expected output should be
```commandline
An example using Bagging with Decision Tree on Iris Dataset
Final Predictions:
[0, 2, 0, 1, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 2, 0], shape=[30], strides=[1], layout=CFcf (0xf), const ndim=1

classes    | 0          | 1          | 2
0          | 11         | 0          | 0
1          | 0          | 7          | 1
2          | 0          | 1          | 10

Test accuracy: 93.333336
 with default Decision Tree params,
 Ensemble Size: 100,
 Bootstrap Proportion: 0.7
 Feature selection proportion: 1

An example using a Random Forest on Iris Dataset
Final Predictions:
[0, 1, 0, 1, 1, 2, 2, 1, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 2, 0], shape=[30], strides=[1], layout=CFcf (0xf), const ndim=1

classes    | 0          | 1          | 2
0          | 11         | 0          | 0
1          | 0          | 8          | 0
2          | 0          | 1          | 10

Test accuracy: 96.666664
 with default Decision Tree params,
 Ensemble Size: 100,
 Bootstrap Proportion: 0.7
 Feature selection proportion: 0.2
```

