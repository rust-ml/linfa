//! Random Forest

use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;
use linfa_trees::{DecisionTree, NodeIter, TreeNode};
use linfa::prelude::*;

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use linfa::dataset::AsSingleTargets;
use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};

use linfa::{
    dataset::{Labels, Records},
    error::Error,
    error::Result,
    traits::*,
    DatasetBase, Float, Label,
};


#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};


fn main() {
  let mut rng = SmallRng::seed_from_u64(42);

  let (train, test) = linfa_datasets::iris()
  .shuffle(&mut rng)
  .split_with_ratio(0.8);

  train.bootstrap((10, 10), &mut rng);
}
