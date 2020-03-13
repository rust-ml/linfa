use crate::decision_trees::hyperparameters::DecisionTreeParams;
use ndarray::{s, stack, Array, Array1, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_rand::rand::Rng;

struct TreeNode {
    feature_idx: usize,
    split_value: f64,
    left_child: Option<Box<TreeNode>>,
    right_child: Option<Box<TreeNode>>,
}

impl TreeNode {
    fn fit(
        x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
        row_idxs: &Vec<usize>,
    ) -> Self {
        let n_classes = x.ncols() as u64;

        let mut best_feature_idx = None;
        let mut best_split_value = None;
        let mut best_score = None;

        // 1. Find best split for current level
        for feature_idx in 0..(x.ncols()) {
            println!("Looking at feature {}", feature_idx);

            let split_values = split_values_for_feature(&x, feature_idx);
            println!("Could split on any of {:?}", split_values);

            for sv in split_values {
                let (left_idxs, right_idxs) = split_on_feature_by_value(&x, feature_idx, sv);
                let left_score = gini_impurity(&y, &left_idxs, n_classes);
                let right_score = gini_impurity(&y, &right_idxs, n_classes);
                let score = (left_score + right_score) / 2.0;

                if best_score.is_none() || score < best_score.unwrap() {
                    best_feature_idx = Some(feature_idx);
                    best_split_value = Some(sv);
                    best_score = Some(score);
                }
            }
        }

        let best_feature_idx = best_feature_idx.unwrap();
        let best_split_value = best_split_value.unwrap();

        // 2. Obtain splitted datasets

        let (left_idxs, right_idxs) =
            split_on_feature_by_value(&x, best_feature_idx, best_split_value);

        // 3. Recurse and refit on splitted data

        let left_child = match left_idxs.len() {
            l if l > 0 => Some(Box::new(TreeNode::fit(&x, &y, &left_idxs))),
            _ => None,
        };

        let right_child = match right_idxs.len() {
            l if l > 0 => Some(Box::new(TreeNode::fit(&x, &y, &right_idxs))),
            _ => None,
        };

        TreeNode {
            feature_idx: best_feature_idx,
            split_value: best_split_value,
            left_child: left_child,
            right_child: right_child,
        }
    }
}

pub struct DecisionTree {
    hyperparameters: DecisionTreeParams,
    root_node: TreeNode,
}

impl DecisionTree {
    pub fn fit(
        hyperparameters: DecisionTreeParams,
        x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
        rng: &mut impl Rng,
    ) -> Self {
        let all_idxs = 0..(x.nrows());
        let root_node = TreeNode::fit(&x, &y, &all_idxs.collect());

        Self {
            hyperparameters,
            root_node,
        }
    }

    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<u64> {
        Array1::ones(5)
    }

    pub fn hyperparameters(&self) -> &DecisionTreeParams {
        &self.hyperparameters
    }
}

pub fn split_on_feature_by_value(
    x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    feature_idx: usize,
    split_value: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = vec![];
    let mut right = vec![];

    for (idx, row) in x.genrows().into_iter().enumerate() {
        if row[feature_idx] < split_value {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    (left, right)
}

fn split_values_for_feature(
    x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    feature_idx: usize,
) -> Vec<f64> {
    let mut values = x.slice(s![.., feature_idx]).to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));

    let mut split_values = vec![];
    let mut last_value = None;

    for v in values {
        if let Some(lv) = last_value {
            if lv != v {
                split_values.push((lv + v) / 2.0)
            }
        }
        last_value = Some(v);
    }

    split_values
}

fn gini_impurity(
    labels: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
    row_idxs: &Vec<usize>,
    n_classes: u64,
) -> f64 {
    let n_samples = row_idxs.len();
    assert!(n_samples > 0);

    let mut class_freq = vec![0.0; n_classes as usize];

    let mut row_idxs_head = 0;
    let mut next_row_idx = row_idxs.get(row_idxs_head);

    for (idx, label) in labels.iter().enumerate() {
        if let Some(nri) = next_row_idx {
            if &idx == nri {
                class_freq[*label as usize] += 1.0;
                row_idxs_head += 1;
                next_row_idx = row_idxs.get(row_idxs_head);
            }
        } else {
            break;
        }
    }

    let purity: f64 = class_freq
        .iter()
        .map(|x| x / (n_samples as f64))
        .map(|x| x * x)
        .sum();

    1.0 - purity
}

fn information_gain(
    parent_labels: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
    child_labels: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
    n_classes: u64,
) -> f64 {
    entropy(&parent_labels, n_classes) - entropy(&child_labels, n_classes)
}

fn entropy(labels: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>, n_classes: u64) -> f64 {
    let n_samples = labels.len();
    let mut class_freq = vec![0.0; n_classes as usize];

    for label in labels.iter() {
        class_freq[*label as usize] += 1.0;
    }

    class_freq
        .iter()
        .map(|x| x / (n_samples as f64))
        .map(|x| if x > 0.0 { -x * x.log2() } else { 0.0 })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn gini_impurity_example() {
        let n_classes = 3;
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);
        let all_idxs = (0..8).collect();

        // Class 0 occurs 75% of the time
        // Class 1 occurs 25% of the time
        // Class 2 occurs 0% of the time
        // Gini impurity is 1 - 0.75*0.75 - 0.25*0.25 - 0*0 = 0.375
        assert_eq!(gini_impurity(&labels, &all_idxs, n_classes), 0.375);
    }

    #[test]
    fn entropy_example() {
        let n_classes = 3;
        let labels = Array::from(vec![0, 0, 0, 0, 0, 0, 1, 1]);

        // Class 0 occurs 75% of the time
        // Class 1 occurs 25% of the time
        // Class 2 occurs 0% of the time
        // Entropy is -0.75*log2(0.75) - 0.25*log2(0.25) - 0*log2(0) = 0.81127812
        assert_eq!(entropy(&labels, n_classes), 0.8112781244591328);

        // If split is perfect then entropy is zero
        let perfect_labels = Array::from(vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(entropy(&perfect_labels, n_classes), 0.0);
    }

    #[test]
    fn information_gain_example() {
        let n_classes = 3;
        let parent_labels = Array::from(vec![0, 0, 0, 0, 0, 1, 1, 2]);
        let child_labels = Array::from(vec![0, 0, 0, 0, 0, 0, 0, 1]);

        // Information gain is just the decrease in entropy from parent to child
        assert_eq!(
            information_gain(&parent_labels, &child_labels, n_classes),
            0.7552304974958021
        );
    }
}
