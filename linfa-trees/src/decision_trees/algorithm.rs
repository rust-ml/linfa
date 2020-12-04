//! Linear decision trees
//!
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::decision_trees::hyperparameters::{DecisionTreeParams, SplitQuality};
use linfa::{
    dataset::{Labels, Records},
    traits::*,
    Dataset, Float, Label,
};
use ndarray::{ArrayBase, Axis, Data, Ix1, Ix2};

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
    fn empty_leaf(prediction: L) -> Self {
        TreeNode {
            feature_idx: 0,
            split_value: F::zero(),
            impurity_decrease: F::zero(),
            left_child: None,
            right_child: None,
            leaf_node: true,
            prediction,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.leaf_node
    }

    fn fit<D: Data<Elem = F>, T: Labels<Elem = L>>(
        data: &Dataset<ArrayBase<D, Ix2>, T>,
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
                .map(|max_depth| depth > max_depth)
                .unwrap_or(false)
        {
            return Self::empty_leaf(prediction);
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
            return Self::empty_leaf(prediction);
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
        }
    }
}

/// A fitted decision tree model.
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
    fn fit(&self, dataset: &Dataset<ArrayBase<D, Ix2>, T>) -> Self::Object {
        let x = dataset.records();
        let all_idxs = RowMask::all(x.nrows());
        let sorted_indices: Vec<_> = (0..(x.ncols()))
            .map(|feature_idx| SortedIndex::of_array_column(&x, feature_idx))
            .collect();

        let root_node = TreeNode::fit(&dataset, &all_idxs, &self, &sorted_indices, 0);

        DecisionTree {
            root_node,
            num_features: dataset.observations(),
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
    pub fn params(n_classes: usize) -> DecisionTreeParams<F, L> {
        DecisionTreeParams {
            n_classes,
            split_quality: SplitQuality::Gini,
            max_depth: None,
            min_weight_split: 2.0,
            min_weight_leaf: 1.0,
            min_impurity_decrease: F::from(0.00001).unwrap(),
            phantom: PhantomData,
        }
    }

    /// Return features_idx of this tree (BFT)
    ///
    pub fn features(&self) -> Vec<usize> {
        // features visited and counted
        let mut visited: HashSet<TreeNode<F, L>> = HashSet::new();
        // queue of nodes yet to explore
        let mut queue = vec![&self.root_node];
        // vector of feature indexes to return
        let mut fitted_features: Vec<usize> = vec![];

        while let Some(node) = queue.pop() {
            // count only internal nodes (where features are)
            if !node.leaf_node {
                // add feature index to list of used features
                fitted_features.push(node.feature_idx);
            }

            // get children and enque them
            let lc = match &node.left_child {
                Some(child) => Some(child),
                _ => None,
            };
            let rc = match &node.right_child {
                Some(child) => Some(child),
                _ => None,
            };
            let children = vec![lc, rc];
            for child in children {
                // extract TreeNode if any
                if let Some(node) = child {
                    if !visited.contains(&node) {
                        visited.insert(*node.clone());
                        queue.push(&node);
                    }
                }
            }
        }

        fitted_features
    }

    /// Return the relative impurity decrease for each feature
    pub fn impurity_decrease(&self) -> Vec<F> {
        // total impurity decrease for each feature
        let mut impurity_decrease = vec![F::zero(); self.num_features];
        // queue of nodes yet to explore
        let mut queue = vec![&self.root_node];
        // total impurity decrease
        let mut total_impurity_decrease = F::zero();

        while let Some(node) = queue.pop() {
            // count only internal nodes (where features are)
            if !node.leaf_node {
                // add feature impurity decrease to list
                impurity_decrease[node.feature_idx] += node.impurity_decrease;
                total_impurity_decrease += node.impurity_decrease;
            }

            if let Some(child) = &node.left_child {
                queue.push(child);
            }

            if let Some(child) = &node.right_child {
                queue.push(child);
            }
        }

        impurity_decrease
            .into_iter()
            .map(|x| x / total_impurity_decrease)
            .collect()
    }

    /// Return root node of the tree
    pub fn root_node(&self) -> &TreeNode<F, L> {
        &self.root_node
    }

    /// Return max depth of the tree
    pub fn max_depth(&self) -> usize {
        // queue of nodes yet to explore
        let mut queue = vec![(0usize, &self.root_node)];
        // max depth, i.e. maximal distance from root to leaf in the current tree
        let mut max_depth = 0;

        while let Some((current_depth, node)) = queue.pop() {
            max_depth = usize::max(max_depth, current_depth);

            if let Some(child) = &node.left_child {
                queue.push((current_depth + 1, &child));
            }

            if let Some(child) = &node.right_child {
                queue.push((current_depth + 1, &child));
            }
        }

        max_depth
    }

    pub fn num_leaves(&self) -> usize {
        // queue of nodes yet to explore
        let mut queue = vec![(0usize, &self.root_node)];
        let mut num_leaves = 0;

        while let Some((current_depth, node)) = queue.pop() {
            if node.is_leaf() {
                num_leaves += 1;
            }

            if let Some(child) = &node.left_child {
                queue.push((current_depth + 1, &child));
            }

            if let Some(child) = &node.right_child {
                queue.push((current_depth + 1, &child));
            }
        }

        num_leaves
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
    use ndarray::{array, s, Array, Array1};

    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn prediction_for_rows_example() {
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);
        let row_mask = RowMask::all(labels.len());

        let dataset = Dataset::new((), labels);
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

        let dataset = Dataset::new(data, targets);

        let model = DecisionTree::params(2).max_depth(Some(2)).fit(&dataset);

        assert_eq!(&model.features(), &[8]);
    }

    #[test]
    /// Small perfectly separable dataset test
    ///
    /// This dataset of three elements is perfectly using the second feature.
    fn perfectly_separable_small() {
        let data = array![[1., 2., 3.5], [1., 2., 3.5], [1., 3., 3.5]];
        let targets = array![0, 0, 1];

        let dataset = Dataset::new(data.clone(), targets);
        let model = DecisionTree::params(2).max_depth(Some(1)).fit(&dataset);

        assert_eq!(&model.predict(data.clone()), &[0, 0, 1]);
    }
}
