# Decision tree learning

`linfa-trees` provides methods for decision tree learning algorithms.

## The Big Picture

`linfa-trees` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

## Current state

`linfa-trees` currently provides an implementation of single tree fitting.

## Random Forest Classifier

An ensemble of decision trees trained on **bootstrapped** subsets of the data **and** **feature-subsampled** per tree. Predictions are made by majority voting across all trees, which typically improves generalization and robustness over a single tree.

**Key features:**
- Configurable number of trees (`n_trees`)
- Optional maximum depth (`max_depth`)
- Fraction of features sampled per tree (`feature_subsample`)
- Reproducible results via RNG seed (`seed`)
- Implements `Fit` and `Predict` traits for seamless integration

### Random Forest Example

```rust
use linfa::prelude::*;
use linfa_datasets::iris;
use linfa_trees::RandomForestParams;
use rand::thread_rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load and split the Iris dataset
    let (train, valid) = iris()
        .shuffle(&mut thread_rng())
        .split_with_ratio(0.8);

    // Train a Random Forest with 100 trees, depth 10, and 80% feature sampling
    let model = RandomForestParams::new(100)
        .max_depth(Some(10))
        .feature_subsample(0.8)
        .seed(42)
        .fit(&train)?;

    // Predict and evaluate
    let preds = model.predict(valid.records.clone());
    let cm = preds.confusion_matrix(&valid)?;
    println!("Accuracy: {:.2}", cm.accuracy());
    Ok(())
}
Run this example with:

bash
Copy
Edit
cargo run --release --example iris_random_forest
Examples
There is an example in the examples/ directory showing how to use decision trees. To run, use:

```bash
cargo run --release --example decision_tree
```

This generates the following tree:

<p align="center"> <img src="./iris-decisiontree.svg" alt="Iris decision tree"> </p>
License
Dual‐licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.

Copy
Edit
