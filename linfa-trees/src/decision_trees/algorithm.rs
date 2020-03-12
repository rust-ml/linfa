use crate::decision_trees::hyperparameters::DecisionTreeParams;
use ndarray::{s, Array, Array1, ArrayBase, Data, Ix1, Ix2};
use ndarray_rand::rand::Rng;

pub struct DecisionTree {
    hyperparameters: DecisionTreeParams,
    // tree: TreeNode,
}

impl DecisionTree {
    pub fn fit(
        hyperparameters: DecisionTreeParams,
        x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
        rng: &mut impl Rng,
    ) -> Self {
        Self { hyperparameters }
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
) -> (
    ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
) {
    let n_rows = x.shape()[0];
    let n_features = x.shape()[1];

    let mut left = vec![];
    let mut left_size = 0;
    let mut right = vec![];
    let mut right_size = 0;

    for row in x.genrows() {
        let mut row = row.to_vec();
        if row[feature_idx] < split_value {
            left.append(&mut row);
            left_size += 1;
        } else {
            right.append(&mut row);
            right_size += 1;
        }
    }

    let left_of_split = Array::from_shape_vec((left_size, n_features), left).unwrap();
    let right_of_split = Array::from_shape_vec((right_size, n_features), right).unwrap();

    (left_of_split, right_of_split)
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
        match last_value {
            Some(lv) => {
                if lv != v {
                    split_values.push((lv + v) / 2.0);
                }
            }
            None => {}
        }

        last_value = Some(v);
    }

    split_values
}

fn gini_impurity(labels: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>, n_classes: u64) -> f64 {
    let n_samples = labels.len();
    let mut class_freq = vec![0.0; n_classes as usize];

    for label in labels.iter() {
        class_freq[*label as usize] += 1.0;
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

        // Class 0 occurs 75% of the time
        // Class 1 occurs 25% of the time
        // Class 2 occurs 0% of the time
        // Gini impurity is 1 - 0.75*0.75 - 0.25*0.25 - 0*0 = 0.375
        assert_eq!(gini_impurity(&labels, n_classes), 0.375);
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
