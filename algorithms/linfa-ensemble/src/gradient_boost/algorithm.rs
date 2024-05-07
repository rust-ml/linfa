use crate::random_forest::DecisionTreeRegressor;
use ndarray::{Array1, Array2};

/*
Source of Algorithm implemented is taken from the blog:
https://en.wikipedia.org/wiki/Gradient_boosting
https://lewtun.github.io/hepml/lesson05_gradient-boosting-deep-dive/
*/

pub struct GBRegressor {
    trees: Vec<DecisionTreeRegressor>,
    num_trees: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_split: usize,
    init_train_target_mean: f64,
}

impl GBRegressor {
    pub fn new(
        num_trees: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
    ) -> Self {
        GBRegressor {
            trees: Vec::new(),
            num_trees,
            learning_rate,
            max_depth,
            min_samples_split,
            init_train_target_mean: 0.0,
        }
    }

    pub fn fit(&mut self, features: &Array2<f64>, targets: &Array1<f64>) {
        self.init_train_target_mean = targets.mean().unwrap_or(0.0);

        let mut y_pred = vec![self.init_train_target_mean; targets.dim()];

        for idx in 0..self.num_trees {
            let gradient = self.gradient(&y_pred, targets);
            // let tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split);
            // tree.fit(&features, &Array1::from_vec(gradient));
            // self.trees.push(tree);

            let tree = self.create_and_train_tree(features, &Array1::from_vec(gradient));
            self.trees.push(tree);

            let update: Array1<f64> = self.trees[idx].predict(features);
            y_pred = self.update_preds(&update, &y_pred);
        }
    }

    fn gradient(&self, y_pred: &Vec<f64>, targets: &Array1<f64>) -> Vec<f64> {
        y_pred
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| (target - pred))
            .collect()
    }

    fn update_preds(&self, update_vals: &Array1<f64>, y_pred: &Vec<f64>) -> Vec<f64> {
        y_pred
            .iter()
            .zip(update_vals.iter())
            .map(|(pred, val)| pred + (self.learning_rate * val))
            .collect()
    }

    fn create_and_train_tree(
        &mut self,
        features: &Array2<f64>,
        targets: &Array1<f64>,
    ) -> DecisionTreeRegressor {
        let mut tree = DecisionTreeRegressor::new(self.max_depth, self.min_samples_split);
        tree.fit(&features, &targets);
        tree
    }

    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        let mut predictions: Vec<Array1<f64>> = Vec::new();
        for tree in &self.trees {
            let prediction = tree.predict(features) * self.learning_rate;
            predictions.push(prediction);
        }

        let num_samples = features.nrows();
        let mut final_predictions = Array1::<f64>::zeros(num_samples);
        final_predictions.fill(self.init_train_target_mean);

        for prediction in predictions {
            final_predictions += &prediction;
        }
        final_predictions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2}; // For floating-point assertions

    fn load_iris_data() -> (Array2<f64>, Array1<f64>) {
        // Load the dataset
        let dataset = linfa_datasets::iris();

        // Extract features; assuming all rows and all but the last column if last is target
        let features = dataset.records().clone();

        // Assuming the target are the labels, we need to convert them or use them as is depending on the use case.
        // If you need to predict a feature, then split differently as shown in previous messages.
        // Here we just clone the labels for the demonstration.
        let targets = dataset.targets().mapv(|x| x as f64);

        (features, targets)
    }

    #[test]
    fn test_gradient_boost_with_iris() {
        let (features, targets) = load_iris_data();

        let mut gb_regressor = GBRegressor::new(50, 0.1, 10, 3);
        gb_regressor.fit(&features, &targets);
        let predictions = gb_regressor.predict(&features);

        // Define a tolerance level
        let tolerance = 0.01; // Tolerance level for correct classification
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
