use crate::decision_trees::hyperparameters::{DecisionTreeParams, SplitQuality};
use ndarray::{s, Array, Array1, ArrayBase, Data, Ix1, Ix2};

struct SplitCandidate {
    split_value: f64,
    left_class_freq: Vec<u64>,
    right_class_freq: Vec<u64>,
}

struct SplitIterator {
    left_idxs: Vec<usize>,
    right_idxs: Vec<usize>,
    left_class_freq: Vec<u64>,
    right_class_freq: Vec<u64>,
    feature: Array1<f64>,
    labels: Array1<u64>,
}

impl SplitIterator {
    fn new(
        feature: Array1<f64>,
        labels: Array1<u64>,
        row_idxs: &Vec<usize>,
        n_classes: u64,
    ) -> SplitIterator {
        let mut feature_pairs = vec![];
        for idx in row_idxs.iter() {
            feature_pairs.push((*idx, &feature[*idx]));
        }

        feature_pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Less));

        let left_idxs = feature_pairs.iter().map(|a| a.0).collect();
        let right_idxs = vec![];

        let left_class_freq = class_frequencies(&labels, &row_idxs, n_classes);
        let right_class_freq = vec![0; n_classes as usize];

        SplitIterator {
            left_idxs: left_idxs,
            right_idxs: right_idxs,
            left_class_freq: left_class_freq,
            right_class_freq: right_class_freq,
            feature: feature,
            labels: labels,
        }
    }
}

impl Iterator for SplitIterator {
    type Item = SplitCandidate;

    fn next(&mut self) -> Option<SplitCandidate> {
        if self.left_idxs.len() < 2 {
            None
        } else {
            self.right_idxs.push(self.left_idxs.pop().unwrap());
            let swapped_class = self.labels[*self.left_idxs.last().unwrap()];
            self.left_class_freq[swapped_class as usize] -= 1;
            self.right_class_freq[swapped_class as usize] += 1;

            let left_idx = self.left_idxs.last().unwrap();
            let right_idx = self.right_idxs.last().unwrap();

            let left_value = self.feature[*left_idx];
            let right_value = self.feature[*right_idx];

            let split_value = (left_value + right_value) as f64 / 2.0;

            Some(SplitCandidate {
                split_value: split_value,
                left_class_freq: self.left_class_freq.to_vec(),
                right_class_freq: self.right_class_freq.to_vec(),
            })
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
        x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
        row_idxs: &Vec<usize>,
        hyperparameters: &DecisionTreeParams,
        depth: u64,
    ) -> Self {
        let mut leaf_node = false;

        leaf_node |= row_idxs.len() < (hyperparameters.min_samples_split as usize);

        if let Some(max_depth) = hyperparameters.max_depth {
            leaf_node |= depth > max_depth;
        }

        let prediction = prediction_for_rows(&y, &row_idxs, hyperparameters.n_classes);

        let mut best_feature_idx = None;
        let mut best_split_value = None;
        let mut best_score = None;

        let parent_class_freq = class_frequencies(&y, row_idxs, hyperparameters.n_classes);

        // 1. Find best split for current level
        for feature_idx in 0..(x.ncols()) {
            let feature = x.slice(s![.., feature_idx]).to_owned();
            let labels = y.to_owned();

            for split_candidate in
                SplitIterator::new(feature, labels, row_idxs, hyperparameters.n_classes)
            {
                let (left_score, right_score) = match hyperparameters.split_quality {
                    SplitQuality::Gini => (
                        -gini_impurity(&split_candidate.left_class_freq),
                        -gini_impurity(&split_candidate.right_class_freq),
                    ),
                    SplitQuality::Entropy => (
                        information_gain(&parent_class_freq, &split_candidate.left_class_freq),
                        information_gain(&parent_class_freq, &split_candidate.right_class_freq),
                    ),
                };

                let left_weight: f64 = split_candidate.left_class_freq.iter().sum::<u64>() as f64
                    / row_idxs.len() as f64;
                let right_weight: f64 = split_candidate.right_class_freq.iter().sum::<u64>() as f64
                    / row_idxs.len() as f64;

                let score = left_weight * left_score + right_weight * right_score;

                if best_score.is_none() || score > best_score.unwrap() {
                    best_feature_idx = Some(feature_idx);
                    best_split_value = Some(split_candidate.split_value);
                    best_score = Some(score);
                }
            }
        }

        leaf_node |= best_score.is_none();

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

        // 2. Obtain splitted datasets

        let (left_idxs, right_idxs) =
            split_on_feature_by_value(&x, &row_idxs, best_feature_idx, best_split_value);

        // 3. Recurse and refit on splitted data

        let left_child = match left_idxs.len() {
            l if l > 0 => Some(Box::new(TreeNode::fit(
                &x,
                &y,
                &left_idxs,
                &hyperparameters,
                depth + 1,
            ))),
            _ => None,
        };

        let right_child = match right_idxs.len() {
            l if l > 0 => Some(Box::new(TreeNode::fit(
                &x,
                &y,
                &right_idxs,
                &hyperparameters,
                depth + 1,
            ))),
            _ => None,
        };

        let leaf_node = left_child.is_none() || right_child.is_none();

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
        x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
    ) -> Self {
        let all_idxs = 0..(x.nrows());
        let root_node = TreeNode::fit(&x, &y, &all_idxs.collect(), &hyperparameters, 0);

        Self {
            hyperparameters,
            root_node,
        }
    }

    /// Make predictions for each row of a matrix of features `x`.
    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<u64> {
        let mut preds = vec![];

        for row in x.genrows().into_iter() {
            preds.push(make_prediction(&row, &self.root_node));
        }

        Array::from(preds)
    }

    pub fn hyperparameters(&self) -> &DecisionTreeParams {
        &self.hyperparameters
    }
}

/// Classify a sample &x recursively using the tree node `node`.
fn make_prediction(x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix1>, node: &TreeNode) -> u64 {
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

fn split_on_feature_by_value(
    x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    row_idxs: &Vec<usize>,
    feature_idx: usize,
    split_value: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = vec![];
    let mut right = vec![];

    for idx in row_idxs.iter() {
        let v = *x.get((*idx, feature_idx)).unwrap();
        if v < split_value {
            left.push(*idx);
        } else {
            right.push(*idx);
        }
    }

    (left, right)
}

/// Given an array of labels and a mask `row_idxs` calculate the frequency of
/// each class from 0 to `n_classes-1`.
fn class_frequencies(
    labels: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
    row_idxs: &Vec<usize>,
    n_classes: u64,
) -> Vec<u64> {
    let n_samples = row_idxs.len();
    assert!(n_samples > 0);

    let mut class_freq = vec![0; n_classes as usize];

    for idx in row_idxs.iter() {
        let label = labels[*idx];
        class_freq[label as usize] += 1;
    }

    class_freq
}

/// Make a point prediction for a subset of rows in the dataset based on the
/// class that occurs the most frequent. If two classes occur with the same
/// frequency then the first class is selected.
fn prediction_for_rows(
    labels: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
    row_idxs: &Vec<usize>,
    n_classes: u64,
) -> u64 {
    let class_freq = class_frequencies(labels, row_idxs, n_classes);

    // Find the class with greatest frequency
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

/// Given an array of labels and a mask `row_idxs` calculate the gini impurity
/// of the subset.
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

/// Given an array of labels and the parent and child mask for before and after
/// a potential split calculate the information gain of making the split.
fn information_gain(parent_class_freq: &Vec<u64>, class_freq: &Vec<u64>) -> f64 {
    entropy(&parent_class_freq) - entropy(&class_freq)
}

/// Given an array of labels and a mask `row_idxs` calculate the entropy of the
/// subset.
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

    #[test]
    fn class_freq_example() {
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);

        assert_eq!(class_frequencies(&labels, &vec![0, 1, 2], 3), vec![3, 0, 0]);
        assert_eq!(class_frequencies(&labels, &vec![0, 6, 7], 3), vec![1, 2, 0]);
    }

    #[test]
    fn prediction_for_rows_example() {
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);
        assert_eq!(prediction_for_rows(&labels, &vec![0, 1, 2, 6, 7], 3), 0);
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

    #[test]
    fn information_gain_example() {
        let parent_class_freq = vec![5, 2, 1];
        let class_freq = vec![5, 2, 0];

        // Information gain is just the decrease in entropy from parent to child
        assert_abs_diff_eq!(
            information_gain(&parent_class_freq, &class_freq),
            0.435674,
            epsilon = 1e-5
        );
    }
}
