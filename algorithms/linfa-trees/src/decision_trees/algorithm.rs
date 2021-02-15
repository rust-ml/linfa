//! Linear decision trees
//!
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use ndarray::{ArrayBase, Axis, Data, Ix1, Ix2};

use super::hyperparameters::{DecisionTreeParams, SplitQuality};
use super::NodeIter;
use super::Tikz;
use linfa::{
    dataset::{Labels, Records},
    traits::*,
    DatasetBase, Float, Label,
};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// RowMask tracks observations
///
/// The decision tree algorithm splits observations at a certain split value for a specific feature. The
/// left and right children can then only use a certain number of observations. In order to track
/// that the observations are masked with a boolean vector, hiding all observations which are not
/// applicable in a lower tree.
struct RowMask {
    mask: Vec<bool>,
    nsamples: usize,
}

impl RowMask {
    fn all(nsamples: usize) -> Self {
        RowMask {
            mask: vec![true; nsamples as usize],
            nsamples,
        }
    }

    fn none(nsamples: usize) -> Self {
        RowMask {
            mask: vec![false; nsamples as usize],
            nsamples: 0,
        }
    }

    fn mark(&mut self, idx: usize) {
        self.mask[idx] = true;
        self.nsamples += 1;
    }
}

/// Sorted values of observations with indices (always for a particular feature)
struct SortedIndex<F: Float> {
    sorted_values: Vec<(usize, F)>,
}

impl<F: Float> SortedIndex<F> {
    fn of_array_column(x: &ArrayBase<impl Data<Elem = F>, Ix2>, feature_idx: usize) -> Self {
        let sliced_column: Vec<F> = x.index_axis(Axis(1), feature_idx).to_vec();
        let mut pairs: Vec<(usize, F)> = sliced_column.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater));

        SortedIndex {
            sorted_values: pairs,
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
        let mut data: Vec<u64> = vec![];
        data.push(self.feature_idx as u64);
        //data.push(self.prediction);
        data.push(self.leaf_node as u64);
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
            split_value: F::zero(),
            impurity_decrease: F::zero(),
            left_child: None,
            right_child: None,
            leaf_node: true,
            prediction,
            depth,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.leaf_node
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn prediction(&self) -> Option<L> {
        if self.is_leaf() {
            return Some(self.prediction.clone());
        } else {
            None
        }
    }

    /// Return both childs
    pub fn childs(&self) -> Vec<&Option<Box<TreeNode<F, L>>>> {
        vec![&self.left_child, &self.right_child]
    }

    /// Return the split and its impurity decrease
    pub fn split(&self) -> (usize, F, F) {
        (self.feature_idx, self.split_value, self.impurity_decrease)
    }

    fn fit<D: Data<Elem = F>, T: Labels<Elem = L>>(
        data: &DatasetBase<ArrayBase<D, Ix2>, T>,
        mask: &RowMask,
        hyperparameters: &DecisionTreeParams<F, L>,
        sorted_indices: &[SortedIndex<F>],
        depth: usize,
    ) -> Self {
        // compute weighted frequencies for target classes
        let parent_class_freq = data.frequencies_with_mask(&mask.mask);
        // set our prediction for this subset to the modal class
        let prediction = find_modal_class(&parent_class_freq);

        // return empty leaf when we don't have enough samples or the maximal depth is reached
        if (mask.nsamples as f32) < hyperparameters.min_weight_split
            || hyperparameters
                .max_depth
                .map(|max_depth| depth >= max_depth)
                .unwrap_or(false)
        {
            return Self::empty_leaf(prediction, depth);
        }

        // Find best split for current level
        let mut best = None;

        // Iterate over features
        for (feature_idx, sorted_index) in sorted_indices.iter().enumerate() {
            let mut left_class_freq = parent_class_freq.clone();
            let mut right_class_freq = HashMap::new();

            // We keep a running total of the aggregate weight in the left split
            // to avoid having to sum over the hash map
            let total_weight = parent_class_freq.values().sum::<f32>();
            let mut weight_on_left_side = total_weight;
            let mut weight_on_right_side = 0.0;

            // Iterate over sorted values
            for i in 0..mask.mask.len() - 1 {
                let (presorted_index, mut split_value) = sorted_index.sorted_values[i];

                if !mask.mask[presorted_index] {
                    continue;
                }

                let sample_class = data.target(presorted_index);
                let sample_weight = data.weight_for(presorted_index);

                // Decrement the weight on the class for this sample on the left
                // side by the weight of this sample
                *left_class_freq.get_mut(sample_class).unwrap() -= sample_weight;
                weight_on_left_side -= sample_weight;

                // Increment the weight on the class for this sample on the
                // right side by the weight of this sample
                *right_class_freq.entry(sample_class).or_insert(0.0) += sample_weight;
                weight_on_right_side += sample_weight;

                // Continue if the next values is equal
                if (sorted_index.sorted_values[i].1 - sorted_index.sorted_values[i + 1].1).abs()
                    < F::from(1e-5).unwrap()
                {
                    continue;
                }

                // If the split would result in too few samples in a leaf
                // then skip computing the quality
                if weight_on_left_side < hyperparameters.min_weight_leaf
                    || weight_on_right_side < hyperparameters.min_weight_leaf
                {
                    continue;
                }

                // Calculate the quality of each resulting subset of the dataset
                let (left_score, right_score) = match hyperparameters.split_quality {
                    SplitQuality::Gini => (
                        gini_impurity(&left_class_freq),
                        gini_impurity(&right_class_freq),
                    ),
                    SplitQuality::Entropy => {
                        (entropy(&left_class_freq), entropy(&right_class_freq))
                    }
                };

                // Weight the qualities based on the number of samples in each subset
                let w = weight_on_left_side / total_weight;
                let score = w * left_score + (1.0 - w) * right_score;

                // Take the midpoint from this value and the next one as split_value
                split_value =
                    (split_value + sorted_index.sorted_values[i + 1].1) / F::from(2.0).unwrap();

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

        let impurity_decrease = if let Some((_, _, best_score)) = best {
            let parent_score = match hyperparameters.split_quality {
                SplitQuality::Gini => gini_impurity(&parent_class_freq),
                SplitQuality::Entropy => entropy(&parent_class_freq),
            };
            let parent_score = F::from(parent_score).unwrap();

            // return empty leaf if impurity has not decreased enough
            parent_score - F::from(best_score).unwrap()
        } else {
            // return zero impurity decrease if we have not found any solution
            F::zero()
        };

        if impurity_decrease < hyperparameters.min_impurity_decrease {
            return Self::empty_leaf(prediction, depth);
        }

        let (best_feature_idx, best_split_value, _) = best.unwrap();

        // determine new masks for the left and right subtrees
        let mut left_mask = RowMask::none(data.observations());
        let mut right_mask = RowMask::none(data.observations());

        for i in 0..data.observations() {
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
                &hyperparameters,
                &sorted_indices,
                depth + 1,
            )))
        } else {
            None
        };

        let right_child = if right_mask.nsamples > 0 {
            Some(Box::new(TreeNode::fit(
                data,
                &right_mask,
                &hyperparameters,
                &sorted_indices,
                depth + 1,
            )))
        } else {
            None
        };

        let leaf_node = left_child.is_none() || right_child.is_none();

        TreeNode {
            feature_idx: best_feature_idx,
            split_value: best_split_value,
            impurity_decrease,
            left_child,
            right_child,
            leaf_node,
            prediction,
            depth,
        }
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

/// A fitted decision tree model.
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

impl<F: Float, L: Label, D: Data<Elem = F>> Predict<ArrayBase<D, Ix2>, Vec<L>>
    for DecisionTree<F, L>
{
    /// Make predictions for each row of a matrix of features `x`.
    fn predict(&self, x: ArrayBase<D, Ix2>) -> Vec<L> {
        x.genrows()
            .into_iter()
            .map(|row| make_prediction(&row, &self.root_node))
            .collect()
    }
}

impl<F: Float, L: Label, D: Data<Elem = F>> Predict<&ArrayBase<D, Ix2>, Vec<L>>
    for DecisionTree<F, L>
{
    fn predict(&self, x: &ArrayBase<D, Ix2>) -> Vec<L> {
        self.predict(x.view())
    }
}

impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, D: Data<Elem = F>, T: Labels<Elem = L>>
    Fit<'a, ArrayBase<D, Ix2>, T> for DecisionTreeParams<F, L>
{
    type Object = DecisionTree<F, L>;

    /// Fit a decision tree using `hyperparamters` on the dataset consisting of
    /// a matrix of features `x` and an array of labels `y`.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Self::Object {
        self.validate().unwrap();

        let x = dataset.records();
        let all_idxs = RowMask::all(x.nrows());
        let sorted_indices: Vec<_> = (0..(x.ncols()))
            .map(|feature_idx| SortedIndex::of_array_column(&x, feature_idx))
            .collect();

        let mut root_node = TreeNode::fit(&dataset, &all_idxs, &self, &sorted_indices, 0);
        root_node.prune();

        DecisionTree {
            root_node,
            num_features: dataset.records().ncols(),
        }
    }
}

impl<F: Float, L: Label + std::fmt::Debug> DecisionTree<F, L> {
    /// Defaults are provided if the optional parameters are not specified:
    /// * `split_quality = SplitQuality::Gini`
    /// * `max_depth = None`
    /// * `min_weight_split = 2.0`
    /// * `min_weight_leaf = 1.0`
    /// * `min_impurity_decrease = 0.00001`
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn params() -> DecisionTreeParams<F, L> {
        DecisionTreeParams {
            split_quality: SplitQuality::Gini,
            max_depth: None,
            min_weight_split: 2.0,
            min_weight_leaf: 1.0,
            min_impurity_decrease: F::from(0.00001).unwrap(),
            phantom: PhantomData,
        }
    }

    /// Create a node iterator
    pub fn iter_nodes<'a>(&'a self) -> NodeIter<'a, F, L> {
        // queue of nodes yet to explore
        let queue = vec![&self.root_node];

        NodeIter::new(queue)
    }

    /// Return features_idx of this tree (BFT)
    ///
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
            .map(|(val, n)| {
                if n == 0 {
                    F::zero()
                } else {
                    val / F::from(n).unwrap()
                }
            })
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

    /// Export to tikz
    pub fn export_to_tikz<'a>(&'a self) -> Tikz<'a, F, L> {
        Tikz::new(&self)
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
fn find_modal_class<L: Label>(class_freq: &HashMap<&L, f32>) -> L {
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
fn gini_impurity<L: Label>(class_freq: &HashMap<&L, f32>) -> f32 {
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
fn entropy<L: Label>(class_freq: &HashMap<&L, f32>) -> f32 {
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
    use linfa::metrics::ToConfusionMatrix;
    use ndarray::{array, s, stack, Array, Array1, Array2, Axis};
    use rand_isaac::Isaac64Rng;

    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};

    #[test]
    fn prediction_for_rows_example() {
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);
        let row_mask = RowMask::all(labels.len());

        let dataset = DatasetBase::new((), labels);
        let class_freq = dataset.frequencies_with_mask(&row_mask.mask);

        assert_eq!(find_modal_class(&class_freq), 0);
    }

    #[test]
    fn gini_impurity_example() {
        let class_freq = vec![(&0, 6.0), (&1, 2.0), (&2, 0.0)].into_iter().collect();

        // Class 0 occurs 75% of the time
        // Class 1 occurs 25% of the time
        // Class 2 occurs 0% of the time
        // Gini impurity is 1 - 0.75*0.75 - 0.25*0.25 - 0*0 = 0.375
        assert_abs_diff_eq!(gini_impurity(&class_freq), 0.375, epsilon = 1e-5);
    }

    #[test]
    fn entropy_example() {
        let class_freq = vec![(&0, 6.0), (&1, 2.0), (&2, 0.0)].into_iter().collect();

        // Class 0 occurs 75% of the time
        // Class 1 occurs 25% of the time
        // Class 2 occurs 0% of the time
        // Entropy is -0.75*log2(0.75) - 0.25*log2(0.25) - 0*log2(0) = 0.81127812
        assert_abs_diff_eq!(entropy(&class_freq), 0.81127, epsilon = 1e-5);

        // If split is perfect then entropy is zero
        let perfect_class_freq = vec![(&0, 8.0), (&1, 0.0), (&2, 0.0)].into_iter().collect();

        assert_abs_diff_eq!(entropy(&perfect_class_freq), 0.0, epsilon = 1e-5);
    }

    #[test]
    /// Single feature test
    ///
    /// Generate a dataset where a single feature perfectly correlates
    /// with the target while the remaining features are random gaussian
    /// noise and do not add any information.
    fn single_feature_random_noise_binary() {
        // generate data with 9 white noise and a single correlated feature
        let mut data = Array::random((50, 10), Uniform::new(-4., 4.));
        data.slice_mut(s![.., 8]).assign(
            &(0..50)
                .map(|x| if x < 25 { 0.0 } else { 1.0 })
                .collect::<Array1<_>>(),
        );

        let targets = (0..50).map(|x| x < 25).collect::<Vec<_>>();
        let dataset = DatasetBase::new(data, targets);

        let model = DecisionTree::params().max_depth(Some(2)).fit(&dataset);

        // we should only use feature index 8 here
        assert_eq!(&model.features(), &[8]);
        assert_eq!(
            &model.feature_importance(),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );

        // check for perfect accuracy
        let cm = model.predict(dataset.records()).confusion_matrix(&dataset);
        assert!(cm.accuracy() == 1.0);
    }

    #[test]
    /// Check that for random data the max depth is used
    fn check_max_depth() {
        let mut rng = Isaac64Rng::seed_from_u64(42);

        // create very sparse data
        let data = Array::random_using((50, 50), Uniform::new(-1., 1.), &mut rng);
        let targets = (0..50).collect::<Vec<_>>();

        let dataset = DatasetBase::new(data, targets);

        // check that the provided depth is actually used
        for max_depth in vec![1, 5, 10, 20] {
            let model = DecisionTree::params()
                .max_depth(Some(max_depth))
                .min_impurity_decrease(1e-10f64)
                .min_weight_split(1e-10)
                .fit(&dataset);
            assert_eq!(model.max_depth(), max_depth);
        }
    }

    #[test]
    /// Small perfectly separable dataset test
    ///
    /// This dataset of three elements is perfectly using the second feature.
    fn perfectly_separable_small() {
        let data = array![[1., 2., 3.], [1., 2., 4.], [1., 3., 3.5]];
        let targets = array![0, 0, 1];

        let dataset = DatasetBase::new(data.clone(), targets);
        let model = DecisionTree::params().max_depth(Some(1)).fit(&dataset);

        assert_eq!(&model.predict(data.clone()), &[0, 0, 1]);
    }

    #[test]
    /// Small toy dataset from scikit-sklearn
    fn toy_dataset() {
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

        let dataset = DatasetBase::new(data, targets);
        let model = DecisionTree::params().fit(&dataset);
        let prediction = model.predict(dataset.records());

        let cm = prediction.confusion_matrix(&dataset);
        assert!(cm.accuracy() > 0.95);
    }

    #[test]
    /// Multilabel classification
    fn multilabel_four_uniform() {
        let mut data = stack(
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
            .collect::<Vec<_>>();

        let dataset = DatasetBase::new(data.clone(), targets);

        let model = DecisionTree::params().fit(&dataset);
        let prediction = model.predict(data);

        let cm = prediction.confusion_matrix(&dataset);
        assert!(cm.accuracy() > 0.99);
    }

    #[test]
    #[should_panic]
    /// Check that a small or negative impurity decrease panics
    fn panic_min_impurity_decrease() {
        DecisionTree::<f64, bool>::params()
            .min_impurity_decrease(0.0)
            .validate()
            .unwrap();
    }
}
