//! Linear decision trees
//!
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::decision_trees::hyperparameters::{DecisionTreeParams, SplitQuality};
use linfa::{dataset::{Labels, Records}, traits::*, Dataset, Float, Label};
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

impl<F: Float, L: Label> TreeNode<F, L> {
    fn empty_leaf(prediction: L) -> Self {
        TreeNode {
            feature_idx: 0,
            split_value: F::zero(),
            left_child: None,
            right_child: None,
            leaf_node: true,
            prediction,
        }
    }

    fn fit<D: Data<Elem = F>, T: Labels<Elem = L>>(
        data: &Dataset<ArrayBase<D, Ix2>, T>,
        mask: &RowMask,
        hyperparameters: &DecisionTreeParams<F, L>,
        sorted_indices: &[SortedIndex<F>],
        depth: usize,
    ) -> Self {
        // count occurences of each target class
        let parent_class_freq = data.frequencies_with_mask(&mask.mask);
        // find class which occures the most
        let prediction = prediction_for_rows(&parent_class_freq);

        // return empty leaf when we don't have enough samples or the maximal depth is reached
        if mask.nsamples < hyperparameters.min_samples_split
            || hyperparameters
                .max_depth
                .map(|max_depth| depth > max_depth)
                .unwrap_or(false)
        {
            return Self::empty_leaf(prediction);
        }

        // Find best split for current level
        let mut best = None;

        // iterate over features
        for (feature_idx, sorted_index) in sorted_indices.iter().enumerate() {
            let mut left_class_freq = parent_class_freq.clone();
            let mut right_class_freq = HashMap::new();
            let mut num_left = data.observations();

            // iterate over sorted values
            for i in 0..mask.mask.len() - 1 {
                let (presorted_index, split_value) = sorted_index.sorted_values[i];

                if !mask.mask[presorted_index] {
                    continue;
                }

                // move the class of the current sample from the left subset to the right
                *left_class_freq
                    .get_mut(data.target(presorted_index))
                    .unwrap() -= data.weight_for(presorted_index);

                *right_class_freq
                    .entry(data.target(presorted_index as usize))
                    .or_insert(0.0) += data.weight_for(presorted_index);
                
                num_left -= 1;

                // when classes get too unbalanced, continue
                if num_left < hyperparameters.min_samples_split
                    || (data.observations() - num_left) < hyperparameters.min_samples_split
                {
                    continue;
                }

                // calculate split quality with given metric
                let (left_score, right_score) = match hyperparameters.split_quality {
                    SplitQuality::Gini => (
                        gini_impurity(&left_class_freq),
                        gini_impurity(&right_class_freq),
                    ),
                    SplitQuality::Entropy => {
                        (entropy(&left_class_freq), entropy(&right_class_freq))
                    }
                };

                // calculate score as weighted sum of left and right weights
                let left_weight: f64 =
                    left_class_freq.values().sum::<f32>() as f64 / mask.mask.len() as f64;
                let right_weight: f64 =
                    right_class_freq.values().sum::<f32>() as f64 / mask.mask.len() as f64;

                let score = left_weight * left_score + right_weight * right_score;

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

        let leaf_node = if let Some((_, _, best_score)) = best {
            let parent_score = match hyperparameters.split_quality {
                SplitQuality::Gini => gini_impurity(&parent_class_freq),
                SplitQuality::Entropy => entropy(&parent_class_freq),
            };
            let parent_score = F::from(parent_score).unwrap();

            // return empty leaf if impurity is not decreased enough
            parent_score - F::from(best_score).unwrap() < hyperparameters.min_impurity_decrease
        } else {
            // return empty leaf if we have not found any solution
            true
        };

        if leaf_node {
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

impl<'a, F: Float, L: Label + 'a, D: Data<Elem = F>, T: Labels<Elem = L>>
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

        let root_node = TreeNode::fit(
            &dataset,
            &all_idxs,
            &self,
            &sorted_indices,
            0,
        );

        DecisionTree { root_node }
    }
}

impl<F: Float, L: Label> DecisionTree<F, L> {
    /// Defaults are provided if the optional parameters are not specified:
    /// * `split_quality = SplitQuality::Gini`
    /// * `max_depth = None`
    /// * `min_samples_split = 2`
    /// * `min_samples_leaf = 1`
    /// * `min_impurity_decrease = 0.00001`
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn params(n_classes: usize) -> DecisionTreeParams<F, L> {
        DecisionTreeParams {
            n_classes,
            split_quality: SplitQuality::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
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
        let mut queue: Vec<&TreeNode<F, L>> = vec![];
        // vector of feature indexes to return
        let mut fitted_features: Vec<usize> = vec![];
        // starting node
        let root = self.root_node.clone();
        queue.push(&root);

        while !queue.is_empty() {
            let s = queue.pop();
            if let Some(node) = s {
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
        }

        fitted_features
    }

    pub fn root_node(&self) -> &TreeNode<F, L> {
        &self.root_node
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

/// Make a point prediction for a subset of rows in the dataset based on the
/// class that occurs the most frequent. If two classes occur with the same
/// frequency then the first class is selected.
fn prediction_for_rows<L: Label>(class_freq: &HashMap<&L, f32>) -> L {
    let val = class_freq
        .into_iter()
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
fn gini_impurity<L: Label>(class_freq: &HashMap<&L, f32>) -> f64 {
    let n_samples = class_freq.values().sum::<f32>();
    assert!(n_samples > 0.0);

    let purity: f64 = class_freq
        .values()
        .map(|x| (*x as f64) / (n_samples as f64))
        .map(|x| x * x)
        .sum();

    1.0 - purity
}

/// Given the class frequencies calculates the entropy of the subset.
fn entropy<L: Label>(class_freq: &HashMap<&L, f32>) -> f64 {
    let n_samples = class_freq.values().sum::<f32>();
    assert!(n_samples > 0.0);

    class_freq
        .values()
        .map(|x| (*x as f64) / (n_samples as f64))
        .map(|x| if x > 0.0 { -x * x.log2() } else { 0.0 })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, Array1, s};

    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn prediction_for_rows_example() {
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);
        let row_mask = RowMask::all(labels.len());

        let dataset = Dataset::new((), labels);
        let class_freq = dataset.frequencies_with_mask(&row_mask.mask);

        assert_eq!(prediction_for_rows(&class_freq), 0);
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
            &(0..50).map(|x| if x < 25 { 0.0 } else { 1.0 }).collect::<Array1<_>>()
        );

        let targets = (0..50).map(|x| x < 25).collect::<Vec<_>>();

        let dataset = Dataset::new(data, targets);

        let model = DecisionTree::params(2)
            .max_depth(Some(2))
            .fit(&dataset);

        assert_eq!(&model.features(), &[8]);
    }
}
