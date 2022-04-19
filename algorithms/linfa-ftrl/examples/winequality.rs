use linfa::dataset::{AsSingleTargets, Records};
use linfa::prelude::*;
use linfa_ftrl::{FollowTheRegularizedLeader, Result};
use rand::{rngs::SmallRng, SeedableRng};

fn main() -> Result<()> {
    // Read the data
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
    println!(
        "valid log loss {:?}",
        val_predictions.log_loss(&valid.as_single_targets().to_vec())?
    );
    Ok(())
}
