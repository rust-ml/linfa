//! Linear decision trees
//!
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};

use super::NodeIter;
use super::Tikz;
use super::{DecisionTreeValidParams, SplitQuality};
use linfa::{
    dataset::{AsTargets, Labels, Records},
    error::Error,
    error::Result,
    traits::*,
    DatasetBase, Float, Label,
};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// RowMask tracks observations
///
/// The decision tree algorithm splits observations at a certain split value for a specific feature. The
/// left and right children can then only use a certain number of observations. In order to track
/// that, the observations are masked with a boolean vector, hiding all observations which are not
/// applicable in a lower tree.
struct RowMask {
    mask: Vec<bool>,
    nsamples: usize,
}

impl RowMask {
    /// Generates a RowMask without hidden observations
    ///
    /// ### Parameters
    ///
    /// * `nsamples`: the total number of observations
    ///
    fn all(nsamples: usize) -> Self {
        RowMask {
            mask: vec![true; nsamples as usize],
            nsamples,
        }
    }

    /// Generates a RowMask where all observations are hidden
    ///
    /// ### Parameters
    ///
    /// * `nsamples`: the total number of observations
    fn none(nsamples: usize) -> Self {
        RowMask {
            mask: vec![false; nsamples as usize],
            nsamples: 0,
        }
    }

    /// Sets the observation at the specified index as visible
    ///
    /// ### Parameters
    ///
    /// * `idx`: the index of the observation to turn visible
    ///
    /// ### Panics
    ///
    /// If `idx` is out of bounds
    ///
    fn mark(&mut self, idx: usize) {
        self.mask[idx] = true;
        self.nsamples += 1;
    }
}

/// Sorted values of observations with indices (always for a particular feature)
struct SortedIndex<'a, F: Float> {
    feature_name: &'a str,
    sorted_values: Vec<(usize, F)>,
}

impl<'a, F: Float> SortedIndex<'a, F> {
    /// Sorts the values of a given feature in ascending order
    ///
    /// ### Parameters
    ///
    /// * `x`: the observations to sort
    /// * `feature_idx`: the index of the feature on whch to sort the data
    /// * `feature_name`: the human readable name of the feature
    ///
    /// ### Returns
    ///
    /// A sorted vector of (index, value) pairs obtained by sorting the observations by
    /// the value of the specified feature.
    fn of_array_column(
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        feature_idx: usize,
        feature_name: &'a str,
    ) -> Self {
        let sliced_column: Vec<F> = x.index_axis(Axis(1), feature_idx).to_vec();
        let mut pairs: Vec<(usize, F)> = sliced_column.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater));

        SortedIndex {
            sorted_values: pairs,
            feature_name,
        }
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
/// A node in the decision tree
pub struct TreeNode<F, L> {
    feature_idx: usize,
    feature_name: String,
    split_value: F,
    impurity_decrease: F,
    left_child: Option<Box<TreeNode<F, L>>>,
    right_child: Option<Box<TreeNode<F, L>>>,
    leaf_node: bool,
    prediction: L,
    depth: usize,
}

impl<F: Float, L: Label> Hash for TreeNode<F, L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let data: Vec<u64> = vec![self.feature_idx as u64, self.leaf_node as u64];
        data.hash(state);
    }
}

impl<F, L> Eq for TreeNode<F, L> {}

impl<F, L> PartialEq for TreeNode<F, L> {
    fn eq(&self, other: &Self) -> bool {
        self.feature_idx == other.feature_idx
    }
}

impl<F: Float, L: Label + std::fmt::Debug> TreeNode<F, L> {
    fn empty_leaf(prediction: L, depth: usize) -> Self {
        TreeNode {
            feature_idx: 0,
            feature_name: "".to_string(),
            split_value: F::zero(),
            impurity_decrease: F::zero(),
            left_child: None,
            right_child: None,
            leaf_node: true,
            prediction,
            depth,
        }
    }

    /// Returns true if the node has no children
    pub fn is_leaf(&self) -> bool {
        self.leaf_node
    }

    /// Returns the depth of the node in the decision tree
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns `Some(prediction)` for leaf nodes and `None` for internal nodes.
    pub fn prediction(&self) -> Option<L> {
        if self.is_leaf() {
            Some(self.prediction.clone())
        } else {
            None
        }
    }

    /// Returns both children, first left then right
    pub fn children(&self) -> Vec<&Option<Box<TreeNode<F, L>>>> {
        vec![&self.left_child, &self.right_child]
    }

    /// Return the split (feature index, value) and its impurity decrease
    pub fn split(&self) -> (usize, F, F) {
        (self.feature_idx, self.split_value, self.impurity_decrease)
    }

    /// Returns the name of the feature used in the split if the node is internal,
    /// `None` otherwise
    pub fn feature_name(&self) -> Option<&String> {
        if self.leaf_node {
            None
        } else {
            Some(&self.feature_name)
        }
    }

    /// Recursively fits the node
    fn fit<D: Data<Elem = F>, T: AsTargets<Elem = L> + Labels<Elem = L>>(
        data: &DatasetBase<ArrayBase<D, Ix2>, T>,
        mask: &RowMask,
        hyperparameters: &DecisionTreeValidParams<F, L>,
        sorted_indices: &[SortedIndex<F>],
        depth: usize,
    ) -> Result<Self> {
        // compute weighted frequencies for target classes
        let parent_class_freq = data.label_frequencies_with_mask(&mask.mask);
        // set our prediction for this subset to the modal class
        let prediction = find_modal_class(&parent_class_freq);
        // get targets from dataset
        let target = data.try_single_target()?;

        // return empty leaf when we don't have enough samples or the maximal depth is reached
        if (mask.nsamples as f32) < hyperparameters.min_weight_split()
            || hyperparameters
                .max_depth()
                .map(|max_depth| depth >= max_depth)
                .unwrap_or(false)
        {
            return Ok(Self::empty_leaf(prediction, depth));
        }

        // Find best split for current level
        let mut best = None;

        // Iterate over all features
        for (feature_idx, sorted_index) in sorted_indices.iter().enumerate() {
            let mut right_class_freq = parent_class_freq.clone();
            let mut left_class_freq = HashMap::new();

            // We keep a running total of the aggregate weight in the right split
            // to avoid having to sum over the hash map
            let total_weight = parent_class_freq.values().sum::<f32>();
            let mut weight_on_right_side = total_weight;
            let mut weight_on_left_side = 0.0;

            // We start by putting all available observations in the right subtree
            // and then move the (sorted by `feature_idx`) observations one by one to
            // the left subtree and evaluate the quality of the resulting split. At each
            // iteration, the obtained split is compared with `best`, in order
            // to find the best possible split.
            // The resulting split will then have the observations with a value of their `feature_idx`
            // feature smaller than the split value in the left subtree and the others still in the right
            // subtree
            for i in 0..mask.mask.len() - 1 {
                // (index of the observation, value of its `feature_idx` feature)
                let (presorted_index, mut split_value) = sorted_index.sorted_values[i];

                // Skip if the observation is unavailable in this subtree
                if !mask.mask[presorted_index] {
                    continue;
                }

                // Target and weight of the current observation
                let sample_class = &target[presorted_index];
                let sample_weight = data.weight_for(presorted_index);

                // Move the observation from the right subtree to the left subtree

                // Decrement the weight on the class for this sample on the right
                // side by the weight of this sample
                *right_class_freq.get_mut(sample_class).unwrap() -= sample_weight;
                weight_on_right_side -= sample_weight;

                // Increment the weight on the class for this sample on the
                // right side by the weight of this sample
                *left_class_freq.entry(sample_class.clone()).or_insert(0.0) += sample_weight;
                weight_on_left_side += sample_weight;

                // Continue if the next value is equal, so that equal values end up in the same subtree
                if (sorted_index.sorted_values[i].1 - sorted_index.sorted_values[i + 1].1).abs()
                    < F::cast(1e-5)
                {
                    continue;
                }

                // If the split would result in too few samples in a leaf
                // then skip computing the quality
                if weight_on_right_side < hyperparameters.min_weight_leaf()
                    || weight_on_left_side < hyperparameters.min_weight_leaf()
                {
                    continue;
                }

                // Calculate the quality of each resulting subset of the dataset
                let (left_score, right_score) = match hyperparameters.split_quality() {
                    SplitQuality::Gini => (
                        gini_impurity(&right_class_freq),
                        gini_impurity(&left_class_freq),
                    ),
                    SplitQuality::Entropy => {
                        (entropy(&right_class_freq), entropy(&left_class_freq))
                    }
                };

                // Weight the qualities based on the number of samples in each subset
                let w = weight_on_right_side / total_weight;
                let score = w * left_score + (1.0 - w) * right_score;

                // Take the midpoint from this value and the next one as split_value
                split_value = (split_value + sorted_index.sorted_values[i + 1].1) / F::cast(2.0);

                // override best indices when score improved
                best = match best.take() {
                    None => Some((feature_idx, split_value, score)),
                    Some((_, _, best_score)) if score < best_score => {
                        Some((feature_idx, split_value, score))
                    }
                    x => x,
                };
            }
        }

        // At this point all possible splits for all possible features have been computed
        // and the best one (if any) is stored in `best`. Now we can compute the
        // impurity decrease as `impurity of the node before splitting - impurity of the split`.
        // If the impurity decrease is above the treshold set in the parameters, then the split is
        // applied and `fit` is recursively called in the two resulting subtrees. If there is no
        // possible split, or if it doesn't bring enough impurity decrease, then the node is set as
        // a leaf node that predicts the most common label in the available observations.

        let impurity_decrease = if let Some((_, _, best_score)) = best {
            let parent_score = match hyperparameters.split_quality() {
                SplitQuality::Gini => gini_impurity(&parent_class_freq),
                SplitQuality::Entropy => entropy(&parent_class_freq),
            };
            let parent_score = F::cast(parent_score);

            // return empty leaf if impurity has not decreased enough
            parent_score - F::cast(best_score)
        } else {
            // return zero impurity decrease if we have not found any solution
            F::zero()
        };

        if impurity_decrease < hyperparameters.min_impurity_decrease() {
            return Ok(Self::empty_leaf(prediction, depth));
        }

        let (best_feature_idx, best_split_value, _) = best.unwrap();

        // determine new masks for the left and right subtrees
        let mut left_mask = RowMask::none(data.nsamples());
        let mut right_mask = RowMask::none(data.nsamples());

        for i in 0..data.nsamples() {
            if mask.mask[i] {
                if data.records()[(i, best_feature_idx)] <= best_split_value {
                    left_mask.mark(i);
                } else {
                    right_mask.mark(i);
                }
            }
        }

        // Recurse and refit on left and right subtrees
        let left_child = if left_mask.nsamples > 0 {
            Some(Box::new(TreeNode::fit(
                data,
                &left_mask,
                hyperparameters,
                sorted_indices,
                depth + 1,
            )?))
        } else {
            None
        };

        let right_child = if right_mask.nsamples > 0 {
            Some(Box::new(TreeNode::fit(
                data,
                &right_mask,
                hyperparameters,
                sorted_indices,
                depth + 1,
            )?))
        } else {
            None
        };

        let leaf_node = left_child.is_none() || right_child.is_none();

        Ok(TreeNode {
            feature_idx: best_feature_idx,
            feature_name: sorted_indices[best_feature_idx].feature_name.to_owned(),
            split_value: best_split_value,
            impurity_decrease,
            left_child,
            right_child,
            leaf_node,
            prediction,
            depth,
        })
    }

    /// Prune tree after fitting it
    ///
    /// This removes parts of the tree which results in the same prediction for
    /// all sub-trees. This is called right after fit to ensure that the tree
    /// is small.
    fn prune(&mut self) -> Option<L> {
        if self.is_leaf() {
            return Some(self.prediction.clone());
        }

        let left = self.left_child.as_mut().and_then(|x| x.prune());
        let right = self.right_child.as_mut().and_then(|x| x.prune());

        match (left, right) {
            (Some(x), Some(y)) => {
                if x == y {
                    self.prediction = x.clone();
                    self.right_child = None;
                    self.left_child = None;
                    self.leaf_node = true;

                    Some(x)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// A fitted decision tree model for classification.
///
/// ### Structure
/// A decision tree structure is a binary tree where:
/// * Each internal node specifies a decision, represented by a choice of a feature and a "split value" such that all observations for which
/// `feature <= split_value` is true fall in the left subtree, while the others fall in the right subtree.
///
/// * leaf nodes make predictions, and their prediction is the most popular label in the node
///
/// ### Algorithm
///
/// Starting with a single root node, decision trees are trained recursively by applying the following rule to every
/// node considered:
///
/// * Find the best split value for each feature of the observations belonging in the node;
/// * Select the feature (and its best split value) that maximizes the quality of the split;
/// * If the score of the split is sufficiently larger than the score of the unsplit node, then two child nodes are generated, the left one
///   containing all observations with `feature <= split value` and the right one containing the rest.
/// * If no suitable split is found, the node is marked as leaf and its prediction is set to be the most common label in the node;
///
/// The [quality score](enum.SplitQuality.html) used can be specified in the [parameters](struct.DecisionTreeParams.html).
///
/// ### Predictions
///
/// To predict the label of a sample, the tree is traversed from the root to a leaf, choosing between left and right children according to
/// the values of the features of the sample. The final prediction for the sample is the prediction of the reached leaf.
///
/// ### Additional constraints
///
/// In order to avoid overfitting the training data, some additional constraints on the quality/quantity of splits can be added to the tree.
/// A description of these additional rules is provided in the [parameters](struct.DecisionTreeParams.html) page.
///
/// ### Example
///
/// Here is an example on how to train a decision tree from its parameters:
///
/// ```rust
///
/// use linfa_trees::DecisionTree;
/// use linfa::prelude::*;
/// use linfa_datasets;
///
/// // Load the dataset
/// let dataset = linfa_datasets::iris();
/// // Fit the tree
/// let tree = DecisionTree::params().fit(&dataset).unwrap();
/// // Get accuracy on training set
/// let accuracy = tree.predict(&dataset).confusion_matrix(&dataset).unwrap().accuracy();
///
/// assert!(accuracy > 0.9);
///
/// ```
///
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
pub struct DecisionTree<F: Float, L: Label> {
    root_node: TreeNode<F, L>,
    num_features: usize,
}

impl<F: Float, L: Label + Default, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
    for DecisionTree<F, L>
{
    /// Make predictions for each row of a matrix of features `x`.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        for (row, target) in x.rows().into_iter().zip(y.iter_mut()) {
            *target = make_prediction(&row, &self.root_node);
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
    for DecisionTreeValidParams<F, L>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = DecisionTree<F, L>;

    /// Fit a decision tree using `hyperparamters` on the dataset consisting of
    /// a matrix of features `x` and an array of labels `y`.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let x = dataset.records();
        let feature_names = dataset.feature_names();
        let all_idxs = RowMask::all(x.nrows());
        let sorted_indices: Vec<_> = (0..(x.ncols()))
            .map(|feature_idx| {
                SortedIndex::of_array_column(x, feature_idx, &feature_names[feature_idx])
            })
            .collect();

        let mut root_node = TreeNode::fit(dataset, &all_idxs, self, &sorted_indices, 0)?;
        root_node.prune();

        Ok(DecisionTree {
            root_node,
            num_features: dataset.records().ncols(),
        })
    }
}

impl<F: Float, L: Label> DecisionTree<F, L> {
    /// Create a node iterator in level-order (BFT)
    pub fn iter_nodes(&self) -> NodeIter<F, L> {
        // queue of nodes yet to explore
        let queue = vec![&self.root_node];

        NodeIter::new(queue)
    }

    /// Return features_idx of this tree (BFT)
    pub fn features(&self) -> Vec<usize> {
        // vector of feature indexes to return
        let mut fitted_features = HashSet::new();

        for node in self.iter_nodes().filter(|node| !node.is_leaf()) {
            if !fitted_features.contains(&node.feature_idx) {
                fitted_features.insert(node.feature_idx);
            }
        }

        fitted_features.into_iter().collect::<Vec<_>>()
    }

    /// Return the mean impurity decrease for each feature
    pub fn mean_impurity_decrease(&self) -> Vec<F> {
        // total impurity decrease for each feature
        let mut impurity_decrease = vec![F::zero(); self.num_features];
        let mut num_nodes = vec![0; self.num_features];

        for node in self.iter_nodes().filter(|node| !node.leaf_node) {
            // add feature impurity decrease to list
            impurity_decrease[node.feature_idx] += node.impurity_decrease;
            num_nodes[node.feature_idx] += 1;
        }

        impurity_decrease
            .into_iter()
            .zip(num_nodes.into_iter())
            .map(|(val, n)| if n == 0 { F::zero() } else { val / F::cast(n) })
            .collect()
    }

    /// Return the relative impurity decrease for each feature
    pub fn relative_impurity_decrease(&self) -> Vec<F> {
        let mean_impurity_decrease = self.mean_impurity_decrease();
        let sum = mean_impurity_decrease.iter().cloned().sum();

        mean_impurity_decrease
            .into_iter()
            .map(|x| x / sum)
            .collect()
    }

    /// Return the feature importance, i.e. the relative impurity decrease, for each feature
    pub fn feature_importance(&self) -> Vec<F> {
        self.relative_impurity_decrease()
    }

    /// Return root node of the tree
    pub fn root_node(&self) -> &TreeNode<F, L> {
        &self.root_node
    }

    /// Return max depth of the tree
    pub fn max_depth(&self) -> usize {
        self.iter_nodes()
            .fold(0, |max, node| usize::max(max, node.depth))
    }

    /// Return the number of leaves in this tree
    pub fn num_leaves(&self) -> usize {
        self.iter_nodes().filter(|node| node.is_leaf()).count()
    }

    /// Generates a [`Tikz`](struct.Tikz.html) structure to print the
    /// fitted tree in Tex using tikz and forest, with the following default parameters:
    ///
    /// * `legend=false`
    /// * `complete=true`
    ///
    pub fn export_to_tikz(&self) -> Tikz<F, L> {
        Tikz::new(self)
    }
}

/// Classify a sample &x recursively using the tree node `node`.
fn make_prediction<F: Float, L: Label>(
    x: &ArrayBase<impl Data<Elem = F>, Ix1>,
    node: &TreeNode<F, L>,
) -> L {
    if node.leaf_node {
        node.prediction.clone()
    } else if x[node.feature_idx] < node.split_value {
        make_prediction(x, node.left_child.as_ref().unwrap())
    } else {
        make_prediction(x, node.right_child.as_ref().unwrap())
    }
}

/// Finds the most frequent class for a hash map of frequencies. If two
/// classes have the same weight then the first class found with that
/// frequency is returned.
fn find_modal_class<L: Label>(class_freq: &HashMap<L, f32>) -> L {
    // TODO: Refactor this with fold_first

    let val = class_freq
        .iter()
        .fold(None, |acc, (idx, freq)| match acc {
            None => Some((idx, freq)),
            Some((_best_idx, best_freq)) => {
                if best_freq > freq {
                    acc
                } else {
                    Some((idx, freq))
                }
            }
        })
        .unwrap()
        .0;

    (*val).clone()
}

/// Given the class frequencies calculates the gini impurity of the subset.
fn gini_impurity<L: Label>(class_freq: &HashMap<L, f32>) -> f32 {
    let n_samples = class_freq.values().sum::<f32>();
    assert!(n_samples > 0.0);

    let purity = class_freq
        .values()
        .map(|x| x / n_samples)
        .map(|x| x * x)
        .sum::<f32>();

    1.0 - purity
}

/// Given the class frequencies calculates the entropy of the subset.
fn entropy<L: Label>(class_freq: &HashMap<L, f32>) -> f32 {
    let n_samples = class_freq.values().sum::<f32>();
    assert!(n_samples > 0.0);

    class_freq
        .values()
        .map(|x| x / n_samples)
        .map(|x| if x > 0.0 { -x * x.log2() } else { 0.0 })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use linfa::{error::Result, metrics::ToConfusionMatrix, Dataset, ParamGuard};
    use ndarray::{array, concatenate, s, Array, Array1, Array2, Axis};
    use rand::rngs::SmallRng;

    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};

    #[test]
    fn prediction_for_rows_example() {
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);
        let row_mask = RowMask::all(labels.len());

        let dataset: DatasetBase<(), Array1<usize>> = DatasetBase::new((), labels);
        let class_freq = dataset.label_frequencies_with_mask(&row_mask.mask);

        assert_eq!(find_modal_class(&class_freq), 0);
    }

    #[test]
    fn gini_impurity_example() {
        let class_freq = vec![(0, 6.0), (1, 2.0), (2, 0.0)].into_iter().collect();

        // Class 0 occurs 75% of the time
        // Class 1 occurs 25% of the time
        // Class 2 occurs 0% of the time
        // Gini impurity is 1 - 0.75*0.75 - 0.25*0.25 - 0*0 = 0.375
        assert_abs_diff_eq!(gini_impurity(&class_freq), 0.375, epsilon = 1e-5);
    }

    #[test]
    fn entropy_example() {
        let class_freq = vec![(0, 6.0), (1, 2.0), (2, 0.0)].into_iter().collect();

        // Class 0 occurs 75% of the time
        // Class 1 occurs 25% of the time
        // Class 2 occurs 0% of the time
        // Entropy is -0.75*log2(0.75) - 0.25*log2(0.25) - 0*log2(0) = 0.81127812
        assert_abs_diff_eq!(entropy(&class_freq), 0.81127, epsilon = 1e-5);

        // If split is perfect then entropy is zero
        let perfect_class_freq = vec![(0, 8.0), (1, 0.0), (2, 0.0)].into_iter().collect();

        assert_abs_diff_eq!(entropy(&perfect_class_freq), 0.0, epsilon = 1e-5);
    }

    #[test]
    /// Single feature test
    ///
    /// Generate a dataset where a single feature perfectly correlates
    /// with the target while the remaining features are random gaussian
    /// noise and do not add any information.
    fn single_feature_random_noise_binary() -> Result<()> {
        // generate data with 9 white noise and a single correlated feature
        let mut data = Array::random((50, 10), Uniform::new(-4., 4.));
        data.slice_mut(s![.., 8]).assign(
            &(0..50)
                .map(|x| if x < 25 { 0.0 } else { 1.0 })
                .collect::<Array1<_>>(),
        );

        let targets = (0..50).map(|x| x < 25).collect::<Array1<_>>();
        let dataset = Dataset::new(data, targets);

        let model = DecisionTree::params().max_depth(Some(2)).fit(&dataset)?;

        // we should only use feature index 8 here
        assert_eq!(&model.features(), &[8]);

        let ground_truth = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        for (imp, truth) in model.feature_importance().iter().zip(&ground_truth) {
            assert_abs_diff_eq!(imp, truth, epsilon = 1e-15);
        }

        // check for perfect accuracy
        let cm = model
            .predict(dataset.records())
            .confusion_matrix(&dataset)?;
        assert_abs_diff_eq!(cm.accuracy(), 1.0, epsilon = 1e-15);

        Ok(())
    }

    #[test]
    /// Check that for random data the max depth is used
    fn check_max_depth() -> Result<()> {
        let mut rng = SmallRng::seed_from_u64(42);

        // create very sparse data
        let data = Array::random_using((50, 50), Uniform::new(-1., 1.), &mut rng);
        let targets = (0..50).collect::<Array1<usize>>();

        let dataset = Dataset::new(data, targets);

        // check that the provided depth is actually used
        for max_depth in &[1, 5, 10, 20] {
            let model = DecisionTree::params()
                .max_depth(Some(*max_depth))
                .min_impurity_decrease(1e-10f64)
                .min_weight_split(1e-10)
                .fit(&dataset)?;
            assert_eq!(model.max_depth(), *max_depth);
        }

        Ok(())
    }

    #[test]
    /// Small perfectly separable dataset test
    ///
    /// This dataset of three elements is perfectly using the second feature.
    fn perfectly_separable_small() -> Result<()> {
        let data = array![[1., 2., 3.], [1., 2., 4.], [1., 3., 3.5]];
        let targets = array![0, 0, 1];

        let dataset = Dataset::new(data.clone(), targets);
        let model = DecisionTree::params().max_depth(Some(1)).fit(&dataset)?;

        assert_eq!(model.predict(&data), array![0, 0, 1]);

        Ok(())
    }

    #[test]
    /// Small toy dataset from scikit-sklearn
    fn toy_dataset() -> Result<()> {
        let data = array![
            [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0, -14.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0,],
            [0.0, 0.0, 5.0, 3.0, 0.0, -4.0, 0.0, 0.0, 1.0, -5.0, 0.2, 0.0, 4.0, 1.0,],
            [-1.0, -1.0, 0.0, 0.0, -4.5, 0.0, 0.0, 2.1, 1.0, 0.0, 0.0, -4.5, 0.0, 1.0,],
            [-1.0, -1.0, 0.0, -1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 1.0,],
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
            [-1.0, -2.0, 0.0, 4.0, -3.0, 10.0, 4.0, 0.0, -3.2, 0.0, 4.0, 3.0, -4.0, 1.0,],
            [2.11, 0.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -3.0, 1.0,],
            [2.11, 0.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.0, 0.0, -2.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.0, 0.0, -2.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 11.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -1.0, 0.0,],
            [2.0, 8.0, 5.0, 1.0, 0.5, -4.0, 10.0, 0.0, 1.0, -5.0, 3.0, 0.0, 2.0, 0.0,],
            [2.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -2.0, 3.0, 0.0, 1.0, 0.0,],
            [2.0, 0.0, 1.0, 2.0, 3.0, -1.0, 10.0, 2.0, 0.0, -1.0, 1.0, 2.0, 2.0, 0.0,],
            [1.0, 1.0, 0.0, 2.0, 2.0, -1.0, 1.0, 2.0, 0.0, -5.0, 1.0, 2.0, 3.0, 0.0,],
            [3.0, 1.0, 0.0, 3.0, 0.0, -4.0, 10.0, 0.0, 1.0, -5.0, 3.0, 0.0, 3.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 1.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -3.0, 1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 1.0, 0.0, 0.0, -3.2, 6.0, 1.5, 1.0, -1.0, -1.0,],
            [2.11, 8.0, -6.0, -0.5, 0.0, 10.0, 0.0, 0.0, -3.2, 6.0, 0.5, 0.0, -1.0, -1.0,],
            [2.0, 0.0, 5.0, 1.0, 0.5, -2.0, 10.0, 0.0, 1.0, -5.0, 3.0, 1.0, 0.0, -1.0,],
            [2.0, 0.0, 1.0, 1.0, 1.0, -2.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0,],
            [2.0, 1.0, 1.0, 1.0, 2.0, -1.0, 10.0, 2.0, 0.0, -1.0, 0.0, 2.0, 1.0, 1.0,],
            [1.0, 1.0, 0.0, 0.0, 1.0, -3.0, 1.0, 2.0, 0.0, -5.0, 1.0, 2.0, 1.0, 1.0,],
            [3.0, 1.0, 0.0, 1.0, 0.0, -4.0, 1.0, 0.0, 1.0, -2.0, 0.0, 0.0, 1.0, 0.0,]
        ];

        let targets = array![1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0];

        let dataset = Dataset::new(data, targets);
        let model = DecisionTree::params().fit(&dataset)?;
        let prediction = model.predict(&dataset);

        let cm = prediction.confusion_matrix(&dataset)?;
        assert!(cm.accuracy() > 0.95);

        Ok(())
    }

    #[test]
    /// Multilabel classification
    fn multilabel_four_uniform() -> Result<()> {
        let mut data = concatenate(
            Axis(0),
            &[Array2::random((40, 2), Uniform::new(-1., 1.)).view()],
        )
        .unwrap();

        data.outer_iter_mut().enumerate().for_each(|(i, mut p)| {
            if i < 10 {
                p += &array![-2., -2.]
            } else if i < 20 {
                p += &array![-2., 2.];
            } else if i < 30 {
                p += &array![2., -2.];
            } else {
                p += &array![2., 2.];
            }
        });

        let targets = (0..40)
            .map(|x| match x {
                x if x < 10 => 0,
                x if x < 20 => 1,
                x if x < 30 => 2,
                _ => 3,
            })
            .collect::<Array1<_>>();

        let dataset = Dataset::new(data.clone(), targets);

        let model = DecisionTree::params().fit(&dataset)?;
        let prediction = model.predict(data);

        let cm = prediction.confusion_matrix(&dataset)?;
        assert!(cm.accuracy() > 0.99);

        Ok(())
    }

    #[test]
    #[should_panic]
    /// Check that a small or negative impurity decrease panics
    fn panic_min_impurity_decrease() {
        DecisionTree::<f64, bool>::params()
            .min_impurity_decrease(0.0)
            .check()
            .unwrap();
    }
}
