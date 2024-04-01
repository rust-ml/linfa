use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;
use linfa_trees::DecisionTree;
use linfa::prelude::*;


fn main() {
  let mut rng = SmallRng::seed_from_u64(42);

  let (train, test) = linfa_datasets::iris()
  .shuffle(&mut rng)
  .split_with_ratio(0.8);
}