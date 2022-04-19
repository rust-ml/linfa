# Follow the regularized leader

`linfa-ftrl` provides a pure Rust implementations of follow the regularized leader, proximal, model.

## The Big Picture

`linfa-ftrl` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

The `linfa-ftrl` crate provides Follow The Regularized Leader - Proximal model with L1 and L2 regularization from Logistic Regression, and primarily used for CTR prediction. It actively stores z and n values, needed to calculate weights.
Without L1 and L2 regularization, it is identical to online gradient descent.


See also:
* [Paper about FTRL](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)

## Examples

There is a usage example in the `examples/` directory. To run, use:

```bash
$ cargo run --features linfa/intel-mkl-system --example ftrl
```

<details>
<summary style="cursor: pointer; display:list-item;">
Show source code
</summary>

```rust
use linfa::prelude::*;
use linfa::dataset::{AsSingleTargets, Records};
use linfa_ftrl::{FollowTheRegularizedLeader, Result};
use rand::{rngs::SmallRng, SeedableRng};

// load Winequality dataset
let (train, valid) = linfa_datasets::winequality()
    .map_targets(|v| if *v > 6 { true } else { false })
    .split_with_ratio(0.9);

let params = FollowTheRegularizedLeader::params()
    .alpha(0.005)
    .beta(1.0)
    .l1_ratio(0.005)
    .l2_ratio(1.0);

let mut model = FollowTheRegularizedLeader::new(&params, train.nfeatures());

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