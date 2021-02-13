use linfa::{
    error::{Error, Result},
    Float, Label,
};
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// The metric used to decide the feature on which to split a node
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
pub enum SplitQuality {
    /// Measures the degree of probability of a randomly chosen point being misclassified, defined as
    /// one minus the sum over all labels of the squared probability of encountering that label.
    /// The Gini index of the root is given by the weighted sum of the indexes of ts two subtrees.
    /// At each step the split is applied to the feature which decreases the most the Gini impurity of the root.
    Gini,
    /// Measures the entropy of a subtree, defined as the sum over all labels of the probability of encountering that label in the
    /// subtree times its logarithm in base two, with negative sign. The entropy of the root minus the weighted sum of the entropy
    /// of its two subtrees defines the "information gain" obtained by applying the split. At each step the split is applied to the
    /// feature with the biggest information gain
    Entropy,
}

/// The set of hyperparameters that can be specified for fitting a
/// [decision tree](struct.DecisionTree.html).
///
/// ### Example
///
/// Here is an example on how to train a decision tree from its hyperparams
///
/// ```rust
///
/// use linfa_trees::DecisionTree;
/// use linfa::prelude::*;
/// use linfa_datasets;
///
/// let dataset = linfa_datasets::iris();
///
/// // Fit the tree
/// let tree = DecisionTree::params().fit(&dataset);
/// // Get accuracy on training set
/// let accuracy = tree.predict(dataset.records()).confusion_matrix(&dataset).accuracy();
///
/// assert!(accuracy > 0.9);
///
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
pub struct DecisionTreeParams<F, L> {
    /// The metric used to decide the feature on which to split a node
    pub split_quality: SplitQuality,
    /// Optional limit to the depth of the decision tree
    pub max_depth: Option<usize>,
    /// Minimum weght of samples required to split a node
    pub min_weight_split: f32,
    /// Minimum weight of samples that a split has to place in each leaf
    pub min_weight_leaf: f32,
    /// Minimum decrease in impurity that a split needs to bring in order for it to be appled
    pub min_impurity_decrease: F,

    pub phantom: PhantomData<L>,
}

impl<F: Float, L: Label> DecisionTreeParams<F, L> {
    /// Sets the metric used to decide the feature on which to split a node
    pub fn split_quality(mut self, split_quality: SplitQuality) -> Self {
        self.split_quality = split_quality;
        self
    }

    /// Sets the optional limit to the depth of the decision tree
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the minimum weight of samples required to split a node
    pub fn min_weight_split(mut self, min_weight_split: f32) -> Self {
        self.min_weight_split = min_weight_split;
        self
    }

    /// Sets the minimum weight of samples that a split has to place in each leaf
    pub fn min_weight_leaf(mut self, min_weight_leaf: f32) -> Self {
        self.min_weight_leaf = min_weight_leaf;
        self
    }

    /// Sets the minimum decrease in impurity that a split needs to bring in order for it to be appled
    pub fn min_impurity_decrease(mut self, min_impurity_decrease: F) -> Self {
        self.min_impurity_decrease = min_impurity_decrease;
        self
    }

    /// Checks the correctness of the hyperparameters
    ///
    /// ### Panics
    ///
    /// If the minimum impurity increase is not greater than zero
    pub fn validate(&self) -> Result<()> {
        if self.min_impurity_decrease < F::epsilon() {
            return Err(Error::Parameters(format!(
                "Minimum impurity decrease should be greater than zero, but was {}",
                self.min_impurity_decrease
            )));
        }

        Ok(())
    }
}
