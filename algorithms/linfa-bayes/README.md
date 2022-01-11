# Naive Bayes

`linfa-bayes` provides pure Rust implementations of Naive Bayes algorithms for the Linfa toolkit.

## The Big Picture

`linfa-bayes` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-bayes` currently provides an implementation of the following methods: 

- Gaussian Naive Bayes ([`GaussianNb`](crate::GaussianNb))
- Multinomial Naive Nayes ([`MultinomialNb`](crate::GaussianNb))

## Examples

You can find examples in the `examples/` directory. To run Gaussian Naive Bayes example, use:

```bash
$ cargo run --example winequality --release
```

<details>
<summary style="cursor: pointer; display:list-item;">
Show source code
</summary>

```rust, no_run
use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::{GaussianNb, Result};

// Read in the dataset and convert targets to binary data
let (train, valid) = linfa_datasets::winequality()
    .map_targets(|x| if *x > 6 { "good" } else { "bad" })
    .split_with_ratio(0.9);

// Train the model
let model = GaussianNb::params().fit(&train)?;

// Predict the validation dataset
let pred = model.predict(&valid);

// Construct confusion matrix
let cm = pred.confusion_matrix(&valid)?;

// classes    | bad        | good      
// bad        | 130        | 12        
// good       | 7          | 10    
//
// accuracy 0.8805031, MCC 0.45080978
println!("{:?}", cm);
println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
# Result::Ok(())
```
</details>

To run Multinomial Naive Bayes example, use:

```bash
$ cargo run --example winequality_multinomial --release
```

<details>
<summary style="cursor: pointer; display:list-item;">
Show source code
</summary>

```rust, no_run
use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::{MultinomialNb, Result};

// Read in the dataset and convert targets to binary data
let (train, valid) = linfa_datasets::winequality()
    .map_targets(|x| if *x > 6 { "good" } else { "bad" })
    .split_with_ratio(0.9);

// Train the model
let model = MultinomialNb::params().fit(&train)?;

// Predict the validation dataset
let pred = model.predict(&valid);

// Construct confusion matrix
let cm = pred.confusion_matrix(&valid)?;
// classes    | bad        | good      
// bad        | 88         | 54        
// good       | 10         | 7         

// accuracy 0.5974843, MCC 0.02000631
println!("{:?}", cm);
println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());
# Result::Ok(())
```
</details>
