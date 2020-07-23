use crate::decision_trees::hyperparameters::{DecisionTreeParams, SplitQuality};
use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};
use std::iter::FromIterator;

/// `RowMask` is used to track which rows are still included up to a particular
/// node in the tree for one particular feature.
struct RowMask {
    mask: Vec<bool>,
    n_samples: u64,
}

impl RowMask {
    fn all(n_samples: u64) -> Self {
        RowMask {
            mask: vec![true; n_samples as usize],
            n_samples: n_samples,
        }
    }
}

struct SortedIndex {
    presorted_indices: Vec<usize>,
    features: Vec<f64>,
}

impl SortedIndex {
    fn of_array_column(x: &ArrayBase<impl Data<Elem = f64>, Ix2>, feature_idx: usize) -> Self {
        let sliced_column: Vec<f64> = x.index_axis(Axis(1), feature_idx).to_vec();
        let mut pairs: Vec<(usize, f64)> = sliced_column.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater));

        SortedIndex {
            presorted_indices: pairs.iter().map(|a| a.0).collect(),
            features: pairs.iter().map(|a| a.1).collect(),
        }
    }
}

struct TreeNode {
    feature_idx: usize,
    split_value: f64,
    left_child: Option<Box<TreeNode>>,
    right_child: Option<Box<TreeNode>>,
    leaf_node: bool,
    prediction: u64,
}

impl TreeNode {
    fn fit(
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64>, Ix1>,
        mask: &RowMask,
        hyperparameters: &DecisionTreeParams,
        sorted_indices: &Vec<SortedIndex>,
        depth: u64,
    ) -> Self {
        let mut leaf_node = false;

        leaf_node |= mask.n_samples < hyperparameters.min_samples_split;

        if let Some(max_depth) = hyperparameters.max_depth {
            leaf_node |= depth > max_depth;
        }

        let parent_class_freq = class_frequencies(&y, mask, hyperparameters.n_classes);
        let prediction = prediction_for_rows(&parent_class_freq);

        let mut best_feature_idx = None;
        let mut best_split_value = None;
        let mut best_score = None;

        // Find best split for current level
        for (feature_idx, sorted_index) in sorted_indices.iter().enumerate() {
            let mut left_class_freq = parent_class_freq.clone();
            let mut right_class_freq = vec![0; hyperparameters.n_classes as usize];

            for i in 0..mask.mask.len() - 1 {
                let split_value = sorted_index.features[i];
                let presorted_index = sorted_index.presorted_indices[i];

                if !mask.mask[presorted_index] {
                    continue;
                }

                // Move the class of the current sample from the left subset to the right
                left_class_freq[y[presorted_index as usize] as usize] -= 1;
                right_class_freq[y[presorted_index as usize] as usize] += 1;

                if left_class_freq.iter().sum::<u64>() < hyperparameters.min_samples_split
                    || right_class_freq.iter().sum::<u64>() < hyperparameters.min_samples_split
                {
                    continue;
                }

                let (left_score, right_score) = match hyperparameters.split_quality {
                    SplitQuality::Gini => (
                        gini_impurity(&left_class_freq),
                        gini_impurity(&right_class_freq),
                    ),
                    SplitQuality::Entropy => {
                        (entropy(&left_class_freq), entropy(&right_class_freq))
                    }
                };

                let left_weight: f64 =
                    left_class_freq.iter().sum::<u64>() as f64 / mask.mask.len() as f64;
                let right_weight: f64 =
                    right_class_freq.iter().sum::<u64>() as f64 / mask.mask.len() as f64;

                let score = left_weight * left_score + right_weight * right_score;

                if best_score.is_none() || score < best_score.unwrap() {
                    best_feature_idx = Some(feature_idx);
                    best_split_value = Some(split_value);
                    best_score = Some(score);
                }
            }
        }

        leaf_node |= best_score.is_none();

        if best_score.is_some() {
            let parent_score = match hyperparameters.split_quality {
                SplitQuality::Gini => gini_impurity(&parent_class_freq),
                SplitQuality::Entropy => entropy(&parent_class_freq),
            };

            leaf_node |= parent_score - best_score.unwrap() < hyperparameters.min_impurity_decrease;
        }

        if leaf_node {
            return TreeNode {
                feature_idx: 0,
                split_value: 0.0,
                left_child: None,
                right_child: None,
                leaf_node: true,
                prediction: prediction,
            };
        }

        let best_feature_idx = best_feature_idx.unwrap();
        let best_split_value = best_split_value.unwrap();

        // Determine new masks for the left and right subtrees
        let mut left_mask = vec![false; x.nrows()];
        let mut left_n_samples = 0;
        let mut right_mask = vec![false; x.nrows()];
        let mut right_n_samples = 0;
        for i in 0..(x.nrows()) {
            if mask.mask[i] {
                if x[[i, best_feature_idx]] < best_split_value {
                    left_mask[i] = true;
                    left_n_samples += 1;
                } else {
                    right_mask[i] = true;
                    right_n_samples += 1;
                }
            }
        }

        let left_mask = RowMask {
            mask: left_mask,
            n_samples: left_n_samples,
        };

        let right_mask = RowMask {
            mask: right_mask,
            n_samples: right_n_samples,
        };

        // Recurse and refit on left and right subtrees
        let left_child = match left_mask.n_samples {
            l if l > 0 => Some(Box::new(TreeNode::fit(
                &x,
                &y,
                &left_mask,
                &hyperparameters,
                &sorted_indices,
                depth + 1,
            ))),
            _ => None,
        };

        let right_child = match right_mask.n_samples {
            l if l > 0 => Some(Box::new(TreeNode::fit(
                &x,
                &y,
                &right_mask,
                &hyperparameters,
                &sorted_indices,
                depth + 1,
            ))),
            _ => None,
        };

        leaf_node |= left_child.is_none() || right_child.is_none();

        TreeNode {
            feature_idx: best_feature_idx,
            split_value: best_split_value,
            left_child: left_child,
            right_child: right_child,
            leaf_node: leaf_node,
            prediction: prediction,
        }
    }
}

/// A fitted decision tree model.
pub struct DecisionTree {
    hyperparameters: DecisionTreeParams,
    root_node: TreeNode,
}

impl DecisionTree {
    /// Fit a decision tree using `hyperparamters` on the dataset consisting of
    /// a matrix of features `x` and an array of labels `y`.
    pub fn fit(
        hyperparameters: DecisionTreeParams,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64>, Ix1>,
    ) -> Self {
        let all_idxs = RowMask::all(x.nrows() as u64);
        let sorted_indices = (0..(x.ncols()))
            .map(|feature_idx| SortedIndex::of_array_column(&x, feature_idx))
            .collect();

        let root_node = TreeNode::fit(&x, &y, &all_idxs, &hyperparameters, &sorted_indices, 0);

        Self {
            hyperparameters,
            root_node,
        }
    }

    /// Make predictions for each row of a matrix of features `x`.
    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<u64> {
        Array1::from_iter(
            x.genrows().into_iter().map(|row| make_prediction(&row, &self.root_node))
        )
    }

    pub fn hyperparameters(&self) -> &DecisionTreeParams {
        &self.hyperparameters
    }
}

/// Classify a sample &x recursively using the tree node `node`.
fn make_prediction(x: &ArrayBase<impl Data<Elem = f64>, Ix1>, node: &TreeNode) -> u64 {
    if node.leaf_node {
        node.prediction
    } else {
        if x[node.feature_idx] < node.split_value {
            make_prediction(x, node.left_child.as_ref().unwrap())
        } else {
            make_prediction(x, node.right_child.as_ref().unwrap())
        }
    }
}

/// Given an array of labels and a row mask `mask` calculate the frequency of
/// each class from 0 to `n_classes-1`.
fn class_frequencies(
    labels: &ArrayBase<impl Data<Elem = u64>, Ix1>,
    mask: &RowMask,
    n_classes: u64,
) -> Vec<u64> {
    let n_samples = mask.n_samples;
    assert!(n_samples > 0);

    let mut class_freq = vec![0; n_classes as usize];

    for (idx, included) in mask.mask.iter().enumerate() {
        if *included {
            class_freq[labels[idx] as usize] += 1;
        }
    }

    class_freq
}

/// Make a point prediction for a subset of rows in the dataset based on the
/// class that occurs the most frequent. If two classes occur with the same
/// frequency then the first class is selected.
fn prediction_for_rows(
    class_freq: &Vec<u64>
) -> u64 {
    class_freq
        .iter()
        .enumerate()
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
        .0 as u64
}

/// Given the class frequencies calculates the gini impurity of the subset.
fn gini_impurity(class_freq: &Vec<u64>) -> f64 {
    let n_samples: u64 = class_freq.iter().sum();
    assert!(n_samples > 0);

    let purity: f64 = class_freq
        .iter()
        .map(|x| (*x as f64) / (n_samples as f64))
        .map(|x| x * x)
        .sum();

    1.0 - purity
}

/// Given the class frequencies calculates the entropy of the subset.
fn entropy(class_freq: &Vec<u64>) -> f64 {
    let n_samples: u64 = class_freq.iter().sum();
    assert!(n_samples > 0);

    class_freq
        .iter()
        .map(|x| (*x as f64) / (n_samples as f64))
        .map(|x| if x > 0.0 { -x * x.log2() } else { 0.0 })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    fn of_vec(mask: Vec<bool>) -> RowMask {
        RowMask {
            n_samples: mask.len() as u64,
            mask: mask,
        }
    }

    #[test]
    fn class_freq_example() {
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);

        assert_eq!(
            class_frequencies(&labels, &RowMask::all(labels.len() as u64), 3),
            vec![6, 2, 0]
        );
        assert_eq!(
            class_frequencies(
                &labels,
                &tests::of_vec(vec![false, false, false, false, false, true, true, true]),
                3
            ),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn prediction_for_rows_example() {
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);
        let row_mask = RowMask::all(labels.len() as u64);
        let n_classes = 3;

        let class_freq = class_frequencies(&labels, &row_mask, n_classes);

        assert_eq!(
            prediction_for_rows(&class_freq),
            0
        );
    }

    #[test]
    fn gini_impurity_example() {
        let class_freq = vec![6, 2, 0];

        // Class 0 occurs 75% of the time
        // Class 1 occurs 25% of the time
        // Class 2 occurs 0% of the time
        // Gini impurity is 1 - 0.75*0.75 - 0.25*0.25 - 0*0 = 0.375
        assert_abs_diff_eq!(gini_impurity(&class_freq), 0.375, epsilon = 1e-5);
    }

    #[test]
    fn entropy_example() {
        let class_freq = vec![6, 2, 0];

        // Class 0 occurs 75% of the time
        // Class 1 occurs 25% of the time
        // Class 2 occurs 0% of the time
        // Entropy is -0.75*log2(0.75) - 0.25*log2(0.25) - 0*log2(0) = 0.81127812
        assert_abs_diff_eq!(entropy(&class_freq), 0.81127, epsilon = 1e-5);

        // If split is perfect then entropy is zero
        let perfect_class_freq = vec![8, 0, 0];
        assert_abs_diff_eq!(entropy(&perfect_class_freq), 0.0, epsilon = 1e-5);
    }
}
