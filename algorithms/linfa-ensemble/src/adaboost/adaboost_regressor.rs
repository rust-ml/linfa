
use std::f64::INFINITY;

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The loss function used to update the weights.
pub enum LossFunction {
    Linear,
    Square,
    Exponential,
}

// use ndarray::{Array1, Array2, Axis/};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeRegressor {
    max_depth: usize,
    min_samples_split: usize,
    tree: Option<TreeNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn fit<S>(&mut self, features: &ArrayBase<S, Ix2>, targets: &Array1<f64>, weights: &Array1<f64>)
    where
        S: ndarray::Data<Elem=f64>,
    {
        self.tree = Some(self.build_tree(features, targets, weights, 0));
    }

    fn build_tree<S>(&self, features: &ArrayBase<S, Ix2>, targets: &Array1<f64>, weights: &Array1<f64>, depth: usize) -> TreeNode
    where
        S: ndarray::Data<Elem=f64>,
    {
        if depth >= self.max_depth || features.nrows() < self.min_samples_split {
            return TreeNode {
                feature: 0,
                value: 0.0,
                left: Box::new(None),
                right: Box::new(None),
                output: Some(Self::weighted_mean(targets, weights)),
            };
        }

        let (best_feature, best_value) = self.best_split(features, targets, weights);
        let (left_idxs, right_idxs) = Self::split_dataset(features, best_feature, best_value);

        TreeNode {
            feature: best_feature,
            value: best_value,
            left: Box::new(Some(self.build_tree(&features.select(Axis(0), &left_idxs), &targets.select(Axis(0), &left_idxs), &weights.select(Axis(0), &left_idxs), depth + 1))),
            right: Box::new(Some(self.build_tree(&features.select(Axis(0), &right_idxs), &targets.select(Axis(0), &right_idxs), &weights.select(Axis(0), &right_idxs), depth + 1))),
            output: None,
        }
    }

    fn weighted_mean(values: &Array1<f64>, weights: &Array1<f64>) -> f64 {
        let sum_weights: f64 = weights.sum();
        let weighted_sum: f64 = values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum();
        weighted_sum / sum_weights
    }

    fn best_split<S>(&self, features: &ArrayBase<S, Ix2>, targets: &Array1<f64>, weights: &Array1<f64>) -> (usize, f64)
    where
        S: Data<Elem=f64>,
    {
        let mut best_feature = 0;
        let mut best_value = 0.0;
        let mut best_mse = INFINITY;

        for feature_idx in 0..features.ncols() {
            // Extract the current feature across all samples
            let feature_values: Vec<f64> = features.column(feature_idx).to_vec();

            // Get unique values sorted for potential splits
            let mut unique_values = feature_values.clone();
            unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique_values.dedup();

            // Evaluate each possible split point
            for window in unique_values.windows(2) {
                let split_value = (window[0] + window[1]) / 2.0;
                
                let (left_idxs, right_idxs) = Self::split_dataset(features, feature_idx, split_value);

                let left_mse = Self::calculate_weighted_mse(
                    &targets.select(Axis(0), &left_idxs),
                    &weights.select(Axis(0), &left_idxs)
                );
                let right_mse = Self::calculate_weighted_mse(
                    &targets.select(Axis(0), &right_idxs),
                    &weights.select(Axis(0), &right_idxs)
                );

                let mse = left_mse + right_mse;

                if mse < best_mse {
                    best_mse = mse;
                    best_feature = feature_idx;
                    best_value = split_value;
                }
            }
        }

        (best_feature, best_value)
    }

    fn calculate_weighted_mse(targets: &Array1<f64>, weights: &Array1<f64>) -> f64 {
        if targets.is_empty() {
            return INFINITY;
        }
        let mean = Self::weighted_mean(targets, weights);
        targets.iter().zip(weights).map(|(&x, &w)| w * (x - mean).powi(2)).sum::<f64>() / weights.sum()
    }

    pub fn predict<S>(&self, features: &ArrayBase<S, Ix2>) -> Array1<f64>
    where
        S: ndarray::Data<Elem=f64>,
    {
        let mut predictions = Array1::<f64>::zeros(features.nrows());
        for (i, feature_row) in features.axis_iter(Axis(0)).enumerate() {
            let mut node = &self.tree;
            while let Some(ref n) = *node {
                if let Some(output) = n.output {
                    predictions[i] = output;
                    break;
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

    fn split_dataset<S>(features: &ArrayBase<S, Ix2>, feature_idx: usize, value: f64) -> (Vec<usize>, Vec<usize>)
    where
        S: ndarray::Data<Elem=f64>,
    {
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
}

/// AdaBoost regressor structure.
pub struct AdaBoostRegressor {
    base_estimators: Vec<DecisionTreeRegressor>,
    estimator_weights: Vec<f64>,
    n_estimators: usize,
    learning_rate: f64,
    rng: StdRng,
    max_depth: usize,
    min_samples_split: usize,
}

impl AdaBoostRegressor {
    /// Creates a new AdaBoost Regressor.
    pub fn new(n_estimators: usize, learning_rate: f64, seed: u64, max_depth: usize, min_samples_split: usize) -> Self {
        AdaBoostRegressor {
            base_estimators: Vec::new(),
            estimator_weights: Vec::new(),
            n_estimators,
            learning_rate,
            rng: StdRng::seed_from_u64(seed),
            max_depth,
            min_samples_split,
        }
    }

    /// Fits the AdaBoost regressor to the training data.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let mut sample_weight = Array1::from_elem(y.len(), 1.0 / y.len() as f64);

        for iboost in 0..self.n_estimators {
            let (sample_weight_updated, estimator_weight, error) = self.boost(iboost, x, y, &sample_weight);
            if let (Some(sw), Some(ew), Some(_)) = (sample_weight_updated, estimator_weight, error) {
                self.base_estimators.push(self.train_estimator(x, y, &sw));
                self.estimator_weights.push(ew);
                sample_weight = sw;
            } else {
                break;
            }
        }
    }

    /// A single boosting iteration.
    fn boost(&mut self, _iboost: usize, x: &Array2<f64>, y: &Array1<f64>, sample_weight: &Array1<f64>) -> (Option<Array1<f64>>, Option<f64>, Option<f64>) {
        let sample_count = x.nrows();
        let rng = &mut self.rng;

        // Safe weighted sampling of the training set with replacement
        let weights_dist = WeightedIndex::new(sample_weight);
        if weights_dist.is_err() {
            return (None, None, None); // Handle possible zero or negative weights gracefully
        }
        let weights_dist = weights_dist.unwrap();
        let bootstrap_idx: Vec<usize> = (0..sample_count).map(|_| rng.sample(&weights_dist)).collect();

        // Prepare the bootstrapped sample
        let x_bootstrap = x.select(Axis(0), &bootstrap_idx);
        let y_bootstrap = y.select(Axis(0), &bootstrap_idx);

        // Initialize and train the estimator on the bootstrapped sample
        let mut estimator = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split);
        estimator.fit(&x_bootstrap, &y_bootstrap, &sample_weight.select(Axis(0),&bootstrap_idx));

        // Obtain a prediction for all samples in the training set
        let y_predict = estimator.predict(x);
        let error_vect = (y - &y_predict).mapv(|e| e.abs());

        // Calculate the weighted error of the estimator
        let mut estimator_error: f64 = 0.0;
        let mut total_weight: f64 = 0.0;
        for (err, &weight) in error_vect.iter().zip(sample_weight.iter()) {
            if *err != 0.0 {  // Only consider errors where predictions were incorrect
                estimator_error += weight * err;
            }
            total_weight += weight;
        }
        estimator_error /= total_weight;

        if estimator_error >= 0.5 {
            // If error is greater or equal to 0.5, stop boosting or ignore the weak learner
            return (None, None, None);
        }

        // Calculate the alpha (estimator weight) using the AdaBoost formula
        let alpha = self.learning_rate * ((1.0 - estimator_error) / estimator_error).ln();

        // Update sample weights; weights are increased for incorrectly predicted instances
        let mut new_sample_weight = sample_weight.clone();
        for (weight, &error) in new_sample_weight.iter_mut().zip(error_vect.iter()) {
            let adjustment = if error > 0.0 { alpha.exp() } else { (-alpha).exp() };
            *weight *= adjustment;
        }

        // Normalize the sample weights
        let sum_weights: f64 = new_sample_weight.sum();
        new_sample_weight.mapv_inplace(|w| w / sum_weights);

        self.base_estimators.push(estimator);
        self.estimator_weights.push(alpha);

        (Some(new_sample_weight), Some(alpha), Some(estimator_error))
    }

    /// Train a base estimator on the bootstrapped dataset.
    fn train_estimator(&self, x: &Array2<f64>, y: &Array1<f64>, weights: &Array1<f64>) -> DecisionTreeRegressor {
        // Initialize a new DecisionTreeRegressor with the configured max_depth and min_samples_split.
        let mut estimator = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split);

        // Train the estimator using the provided datasets and associated weights.
        // Assuming `fit` method of DecisionTreeRegressor can handle weighted training.
        estimator.fit(x, y, weights);

        // Return the trained estimator.
        estimator
    }
    /// Predicts the regression value for given samples.
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut predictions = self.base_estimators.iter().map(|est| est.predict(x)).peekable();
        
        // Check if there are any predictions to sum
        if let Some(first_pred) = predictions.peek() {
            let mut total_predictions = Array1::<f64>::zeros(first_pred.dim());

            // Iterate over each prediction and add it to the total_predictions
            for prediction in predictions {
                total_predictions += &prediction;
            }

            total_predictions
        } else {
            // If no predictions are available, return an array of zeros
            Array1::<f64>::zeros(x.nrows())
        }
    }
}