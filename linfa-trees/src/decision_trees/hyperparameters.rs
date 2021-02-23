use linfa::{
    error::{Error, Result},
    Float, Label,
};
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// The metric used to determine the feature by which a node is split
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
pub enum SplitQuality {
    /// Measures the degree of probability of a randomly chosen point in the subtree being misclassified, defined as
    /// one minus the sum over all labels of the squared probability of encountering that label.
    /// The Gini index of the root is given by the weighted sum of the indexes of its two subtrees.
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
/// ```rust
/// use linfa_trees::{DecisionTree, SplitQuality};
/// use linfa_datasets::iris;
/// use linfa::prelude::*;
///
/// // Initialize the default set of parameters
/// let params = DecisionTree::params();
/// // Set the parameters to the desired values
/// let params = params.split_quality(SplitQuality::Entropy).max_depth(Some(5)).min_weight_leaf(2.);
///
/// // Load the data
/// let (train, val) = linfa_datasets::iris().split_with_ratio(0.9);
/// // Fit the decision tree on the training data
/// let tree = params.fit(&train).unwrap();
/// // Predict on validation and check accuracy
/// let val_accuracy = tree.predict(&val).confusion_matrix(&val).unwrap().accuracy();
/// assert!(val_accuracy > 0.99);
/// ```
///
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
pub struct DecisionTreeParams<F, L> {
    pub split_quality: SplitQuality,
    pub max_depth: Option<usize>,
    pub min_weight_split: f32,
    pub min_weight_leaf: f32,
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

    /// Sets the minimum weight of samples required to split a node.
    ///
    /// If the observations do not have associated weights, this value represents
    /// the minimum number of samples required to split a node.
    pub fn min_weight_split(mut self, min_weight_split: f32) -> Self {
        self.min_weight_split = min_weight_split;
        self
    }

    /// Sets the minimum weight of samples that a split has to place in each leaf
    ///
    /// If the observations do not have associated weights, this value represents
    /// the minimum number of samples that a split has to place in each leaf.
    pub fn min_weight_leaf(mut self, min_weight_leaf: f32) -> Self {
        self.min_weight_leaf = min_weight_leaf;
        self
    }

    /// Sets the minimum decrease in impurity that a split needs to bring in order for it to be applied
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
