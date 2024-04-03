use linfa::{
  error::{Error, Result},
  Float, Label, ParamGuard,
};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

// use crate::RandomForest;
use linfa_trees::DecisionTreeParams;
use std::marker::PhantomData;

pub struct RandomForestParams<F, L> {
  // to-do add all the parameters found in sk-learn's RandomForest
  n_estimators: usize,
  bootstrap: bool,
  max_samples: Option<usize>,
  d_tree_params: DecisionTreeParams<F, L>,
}

impl<F: Float, L: Label> RandomForestParams<F, L> {
  pub fn n_estimators(&self) -> usize {
    self.n_estimators
  }
  pub fn tree_params(&self) -> DecisionTreeParams<F, L> {
    self.d_tree_params.clone()
  }
}

// pub struct RandomForestParams<F, L>(RandomForestValidParams<F, L>);

impl<F: Float, L: Label> RandomForestParams<F, L> {
  pub fn new() -> Self {
    Self{
      n_estimators: 100,
      bootstrap: true,
      max_samples: None,  // to-do sklearn uses the total sample count, find a suitable default value
      d_tree_params: DecisionTreeParams::new(),
    }
  }

  /// Sets the total number of estimators in the forest
  pub fn estimators(mut self, n_estimators: usize) -> Self {
    self.n_estimators = n_estimators;
    self
  }

  /// Set the bootstrap to false/true. If set to true, samples are bootstrapped
  /// when building trees. If False, the whole dataset is used to build each tree.
  pub fn bootstrap(mut self, bootstrap: bool) -> Self {
    self.bootstrap = bootstrap;
    self
  }

  pub fn max_samples(mut self, max_samples: usize) -> Self {
    self.max_samples = Some(max_samples);
    self
  }

  pub fn max_depth( self, max_depth: usize) -> Self {
    self.d_tree_params.clone().max_depth(Some(max_depth));
    self
  }


}

impl<F: Float, L: Label> Default for RandomForestParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}


// impl <F: Float, L: Label> RandomForest<F, L> {
//   // Violates the convention that new should return a value of type `Self`
//   #[allow(clippy::new_ret_no_self)]
//   fn params() -> RandomForestParams<F, L> {
//     RandomForestParams::new()
//   }    
// }