+++
title = "Release 0.3.0"
date = "2021-01-21"
+++

Linfa 0.3.0 concentrates on polishing the existing implementation and adds only three new algorithms to the crowd. A new feature system is introduced, which allows the selection of the BLAS/LAPACK backend in the base-crate. The `Dataset` interface is polished and follows the `ndarray` model more closely. The new `linfa-datasets` crate gives easier access to sample datasets and can be used for testing.
<!-- more -->

# New algorithms
 * Approximated DBSCAN has been added to `linfa-clustering` by [@Sauro98]
 * Gaussian Naive Bayes  has been added to `linfa-bayes` by [@VasanthakumarV]
 * Elastic Net linear regression has been added to `linfa-elasticnet` by [@paulkoerbitz] and [@bytesnake]

# Changes
 * Added benchmark to gaussian mixture models (a3eede55)
 * Fixed bugs in linear decision trees, added generator for TiKZ trees (bfa5aebe7)
 * Implemented serde for all crates behind feature flag (4f0b63bb)
 * Implemented new backend features (7296c9ec4)
 * Introduced `linfa-datasets` for easier testing (3cec12b4f)
 * Rename `Dataset` to `DatasetBase` and introduce `Dataset` and `DatasetView` (21dd579cf)
 * Improve kernel tests and documentation (8e81a6d)

# Example
The following section shows a small example how datasets interact with the training and testing of a Linear Decision Tree. 

You can load a dataset, shuffle it and then split it into training and validation sets:
```rust
// initialize pseudo random number generator with seed 42
let mut rng = Isaac64Rng::seed_from_u64(42);
// load the Iris dataset, shuffle and split with ratio 0.8
let (train, test) = linfa_datasets::iris()
    .shuffle(&mut rng)
    .split_with_ratio(0.8);
```
With the training dataset a linear decision tree model can be trained. Entropy is used as a metric for the optimal split here:
```rust
let entropy_model = DecisionTree::params()
    .split_quality(SplitQuality::Entropy)
    .max_depth(Some(100))
    .min_weight_split(10.0)
    .min_weight_leaf(10.0)
    .fit(&train);
```
The validation dataset is now used to estimate the error. For this the true labels are predicted and then a confusion matrix gives clue about the type of error:
```rust
let cm = entropy_model
    .predict(test.records().view())
    .confusion_matrix(&test);

println!("{:?}", cm);

println!(
    "Test accuracy with Entropy criterion: {:.2}%",
    100.0 * cm.accuracy()
);
```
Finally you can analyze which features were used in the decision and export the whole tree it to a `TeX` file. It will contain a TiKZ tree with information on the splitting decision and impurity improvement:
```rust
let feats = entropy_model.features();
println!("Features trained in this tree {:?}", feats);

let mut tikz = File::create("decision_tree_example.tex").unwrap();
tikz.write(gini_model.export_to_tikz().to_string().as_bytes())
    .unwrap();
```
The whole example can be found in [linfa-trees/examples/decision_tree.rs](https://github.com/rust-ml/linfa/blob/master/linfa-trees/examples/decision_tree.rs).
