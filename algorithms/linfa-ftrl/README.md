# Follow the regularized leader

`linfa-ftrl` provides a pure Rust implementations of follow the regularized leader, proximal, model.

## The Big Picture

`linfa-ftrl` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

The `linfa-ftrl` crate provides Follow The Regularized Leader - Proximal model with L1 and L2 regularization from Logistic Regression, and primarily used for CTR prediction. It actively stores z and n values, needed to calculate weights.
Without L1 and L2 regularization, it is identical to online gradient descent.


See also:
* [Paper about Ftrl](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)

## Examples

There is a usage example in the `examples/` directory. To run, use:

```bash
$ cargo run --example winequality
```

<details>
<summary style="cursor: pointer; display:list-item;">
Show source code
</summary>

```rust
use linfa::prelude::*;
use linfa::dataset::{AsSingleTargets, Records};
use linfa_ftrl::{Ftrl, Result};
use rand::{rngs::SmallRng, SeedableRng};

// load Winequality dataset
let (train, valid) = linfa_datasets::winequality()
    .map_targets(|v| if *v > 6 { true } else { false })
    .split_with_ratio(0.9);

let params = Ftrl::params()
    .alpha(0.005)
    .beta(1.0)
    .l1_ratio(0.005)
    .l2_ratio(1.0);

let valid_params = params.clone().check_unwrap();
let mut model = Ftrl::new(valid_params, train.nfeatures());

// Bootstrap each row from the train dataset to imitate online nature of the data flow
let mut rng = SmallRng::seed_from_u64(42);
let mut row_iter = train.bootstrap_samples(1, &mut rng);
for _ in 0..train.nsamples() {
    let b_dataset = row_iter.next().unwrap();
    model = params.fit_with(Some(model), &b_dataset)?;
}
let val_predictions = model.predict(&valid);
println!("valid log loss {:?}", val_predictions.log_loss(&valid.as_single_targets().to_vec())?);
# Result::Ok(())
```
</details>
