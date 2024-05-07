use ndarray::{Array1, Array2, Axis};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

pub struct DecisionTreeRegressor {
    max_depth: usize,
    min_samples_split: usize,
    tree: Option<TreeNode>,
}

#[derive(Debug)]
pub struct TreeNode {
    feature: usize,
    value: f64,
    left: Box<Option<TreeNode>>,
    right: Box<Option<TreeNode>>,
    output: Option<f64>,
}

impl DecisionTreeRegressor {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        DecisionTreeRegressor {
            max_depth,
            min_samples_split,
            tree: None,
        }
    }

    pub fn fit(&mut self, features: &Array2<f64>, targets: &Array1<f64>) {
        self.tree = Some(self.build_tree(features, targets, 0));
    }

    fn build_tree(&self, features: &Array2<f64>, targets: &Array1<f64>, depth: usize) -> TreeNode {
        if depth >= self.max_depth || features.nrows() < self.min_samples_split {
            return TreeNode {
                feature: 0,
                value: 0.0,
                left: Box::new(None),
                right: Box::new(None),
                output: Some(Self::mean(targets)),
            };
        }

        let (best_feature, best_value) = self.best_split(features, targets);
        let (left_idxs, right_idxs) = Self::split_dataset(features, best_feature, best_value);

        let left_tree = if !left_idxs.is_empty() {
            let left_features = features.select(Axis(0), &left_idxs);
            let left_targets = targets.select(Axis(0), &left_idxs);
            Some(self.build_tree(&left_features, &left_targets, depth + 1))
        } else {
            None
        };

        let right_tree = if !right_idxs.is_empty() {
            let right_features = features.select(Axis(0), &right_idxs);
            let right_targets = targets.select(Axis(0), &right_idxs);
            Some(self.build_tree(&right_features, &right_targets, depth + 1))
        } else {
            None
        };

        TreeNode {
            feature: best_feature,
            value: best_value,
            left: Box::new(left_tree),
            right: Box::new(right_tree),
            output: None, // Output is now only assigned at leaves
        }
    }

    fn best_split(&self, features: &Array2<f64>, targets: &Array1<f64>) -> (usize, f64) {
        let mut best_feature = 0;
        let mut best_value = 0.0;
        let mut best_mse = f64::INFINITY;

        for feature_idx in 0..features.ncols() {
            let mut possible_values = features.column(feature_idx).to_vec();
            possible_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            possible_values.dedup();

            for &value in possible_values.iter() {
                let (left_idxs, right_idxs) = Self::split_dataset(features, feature_idx, value);
                if !left_idxs.is_empty() && !right_idxs.is_empty() {
                    let left_targets = targets.select(Axis(0), &left_idxs);
                    let right_targets = targets.select(Axis(0), &right_idxs);
                    let mse = Self::calculate_mse(&left_targets, &right_targets);
                    println!(
                        "Feature: {},\tValue: {},\tMSE: {:.2}",
                        feature_idx, value, mse
                    ); // Debug statement
                    if mse < best_mse {
                        best_mse = mse;
                        best_feature = feature_idx;
                        best_value = value;
                    }
                }
            }
        }

        (best_feature, best_value)
    }

    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::<f64>::zeros(features.nrows());
        for (i, feature_row) in features.axis_iter(Axis(0)).enumerate() {
            let mut node = &self.tree;
            while let Some(ref n) = *node {
                if let Some(output) = n.output {
                    predictions[i] = output;
                    break; // Break if a leaf node with output is reached.
                }
                node = if feature_row[n.feature] <= n.value {
                    &n.left
                } else {
                    &n.right
                };
            }
        }
        predictions
    }

    fn split_dataset(
        features: &Array2<f64>,
        feature_idx: usize,
        value: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_idxs = Vec::new();
        let mut right_idxs = Vec::new();

        for (idx, feature_row) in features.axis_iter(Axis(0)).enumerate() {
            if feature_row[feature_idx] <= value {
                left_idxs.push(idx);
            } else {
                right_idxs.push(idx);
            }
        }

        (left_idxs, right_idxs)
    }

    fn calculate_mse(left_targets: &Array1<f64>, right_targets: &Array1<f64>) -> f64 {
        let left_mean = Self::mean(left_targets);
        let right_mean = Self::mean(right_targets);
        let left_mse: f64 = left_targets.iter().map(|&x| (x - left_mean).powi(2)).sum();
        let right_mse: f64 = right_targets
            .iter()
            .map(|&x| (x - right_mean).powi(2))
            .sum();
        (left_mse + right_mse) / (left_targets.len() + right_targets.len()) as f64
    }

    fn mean(values: &Array1<f64>) -> f64 {
        values.mean().unwrap_or(0.0)
    }
}
pub struct RandomForestRegressor {
    trees: Vec<DecisionTreeRegressor>,
    num_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    rng: ThreadRng,
}

impl RandomForestRegressor {
    pub fn new(num_trees: usize, max_depth: usize, min_samples_split: usize) -> Self {
        let rng = rand::thread_rng();
        RandomForestRegressor {
            trees: Vec::with_capacity(num_trees),
            num_trees,
            max_depth,
            min_samples_split,
            rng,
        }
    }

    pub fn fit(&mut self, features: &Array2<f64>, targets: &Array1<f64>) {
        for _ in 0..self.num_trees {
            let tree = self.create_and_train_tree(features, targets);
            self.trees.push(tree);
        }
    }

    fn create_and_train_tree(
        &mut self,
        features: &Array2<f64>,
        targets: &Array1<f64>,
    ) -> DecisionTreeRegressor {
        let num_samples = features.nrows();
        let indices: Vec<usize> = (0..num_samples).collect();
        let sampled_indices = indices
            .as_slice()
            .choose_multiple(&mut self.rng, num_samples)
            .cloned()
            .collect::<Vec<usize>>();

        let sampled_features = features.select(Axis(0), &sampled_indices);
        let sampled_targets = targets.select(Axis(0), &sampled_indices);

        let mut tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split);
        tree.fit(&sampled_features, &sampled_targets);
        tree
    }

    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        let mut predictions: Vec<Array1<f64>> = Vec::new();
        for tree in &self.trees {
            predictions.push(tree.predict(features));
        }

        let num_samples = features.nrows();
        let mut final_predictions = Array1::<f64>::zeros(num_samples);
        for prediction in predictions {
            final_predictions += &prediction;
        }
        final_predictions /= self.num_trees as f64;
        final_predictions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use linfa_datasets::iris;
    use ndarray::{Array1, Array2}; // For floating-point assertions

    #[test]
    fn test_decision_tree_regressor() {
        let features = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 4.0, 5.0]).unwrap();
        let targets = Array1::from_vec(vec![1.1, 1.9, 3.9, 5.1]);
        let mut regressor = DecisionTreeRegressor::new(3, 1);

        if features.nrows() != targets.len() {
            panic!("Feature and target count mismatch");
        }

        regressor.fit(&features, &targets);
        let predictions = regressor.predict(&features);
        let rmse = calculate_rmse(&targets, &predictions);

        println!("RMSE: {:?}", rmse);
        assert_relative_eq!(rmse, 0.0, epsilon = 0.1);
    }

    fn calculate_rmse(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
        let errors = actual - predicted;
        let mse = errors.mapv(|e| e.powi(2)).mean().unwrap();
        mse.sqrt()
    }

    fn load_iris_data() -> (Array2<f64>, Array1<f64>) {
        // Load the dataset
        let dataset = iris();

        // Extract features; assuming all rows and all but the last column if last is target
        let features = dataset.records().clone();

        // Assuming the target are the labels, we need to convert them or use them as is depending on the use case.
        // If you need to predict a feature, then split differently as shown in previous messages.
        // Here we just clone the labels for the demonstration.
        let targets = dataset.targets().mapv(|x| x as f64);

        (features, targets)
    }

    #[test]
    fn test_random_forest_with_iris() {
        let (features, targets) = load_iris_data();

        let mut forest = RandomForestRegressor::new(100, 10, 3);
        forest.fit(&features, &targets);
        let predictions = forest.predict(&features);

        // Define a tolerance level
        let tolerance = 0.1; // Tolerance level for correct classification
        let mut correct = 0;
        let mut incorrect = 0;

        // Count correct and incorrect predictions
        for (&actual, &predicted) in targets.iter().zip(predictions.iter()) {
            if (predicted - actual).abs() < tolerance {
                correct += 1;
            } else {
                incorrect += 1;
            }
        }

        println!("Correct predictions: {}", correct);
        println!("Incorrect predictions: {}", incorrect);

        let rmse = (&predictions - &targets)
            .mapv(|a| a.powi(2))
            .mean()
            .unwrap()
            .sqrt();

        println!("RMSE: {:?}", rmse);
    }
}
