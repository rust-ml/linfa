use linfa::{
    error::{Error, Result},
    Float, Label, ParamGuard,
};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

use crate::Adaboost;
use linfa_trees::DecisionTreeParams;
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaboostValidParams<F, L> {
    n_estimators: usize,
    learning_rate: f32,
    d_tree_params: DecisionTreeParams<F, L>,
}

impl<F: Float, L: Label> AdaboostValidParams<F, L> {
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    pub fn d_tree_params(&self) -> DecisionTreeParams<F, L> {
        self.d_tree_params.clone()
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaboostParams<F, L>(AdaboostValidParams<F, L>);

impl<F: Float, L: Label> AdaboostParams<F, L> {
    pub fn new() -> Self {
        Self(AdaboostValidParams {
            learning_rate: 0.5,
            n_estimators: 50,
            d_tree_params: DecisionTreeParams::new()
                .min_weight_leaf(0.00001)
                .min_weight_split(0.00001),
        })
    }

    /// Sets the limit to how many stumps will be created
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.0.n_estimators = n_estimators;
        self
    }

    /// Sets the learning rate
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.0.learning_rate = learning_rate;
        self
    }

    /// Sets the params for the weak learner used in Adaboost
    pub fn d_tree_params(mut self, d_tree_params: DecisionTreeParams<F, L>) -> Self {
        self.0.d_tree_params = d_tree_params;
        self
    }
}

impl<F: Float, L: Label> Default for AdaboostParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, L: Label> Adaboost<F, L> {
    /// Defaults are provided if the optional parameters are not specified:
    /// * `n_estimators = 50`
    /// * `learning_rate = 0.00001`
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn params() -> AdaboostParams<F, L> {
        AdaboostParams::new()
    }
}

impl<F: Float, L: Label> ParamGuard for AdaboostParams<F, L> {
    type Checked = AdaboostValidParams<F, L>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.learning_rate < f32::EPSILON {
            Err(Error::Parameters(format!(
                "Minimum learning rate should be greater than zero, but was {}",
                self.0.learning_rate
            )))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
