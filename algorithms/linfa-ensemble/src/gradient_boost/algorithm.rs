use std::{collections::HashMap, iter::zip};

use linfa_trees::{DecisionTree, DecisionTreeParams};

use linfa::{
    dataset::{AsTargets, Labels},
    error::{Error, Result},
    traits::*,
    DatasetBase, Float, Label,
};

use linfa::dataset::AsSingleTargets;
use ndarray::{Array1, Array2, ArrayBase, Data, DataShared, Ix2};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

use crate::random_forest::DecisionTreeRegressor;

pub struct GBRegressor {
    trees: Vec<DecisionTreeRegressor>,
    num_trees: usize,
    learning_rate: f64,
    max_features: usize, // Maximum number of features to consider for each split
    max_depth: usize,
    min_samples_split: usize,
    init_train_target_mean: f64,
}

impl GBRegressor {
    pub fn new(
        num_trees: usize,
        learning_rate: f64,
        max_features: usize,
        max_depth: usize,
        min_samples_split: usize,
    ) -> Self {
        GBRegressor {
            trees: Vec::new(),
            num_trees,
            learning_rate,
            max_features,
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

    fn loss(&self, y_pred: &Vec<f64>, targets: &Array1<f64>) -> Vec<f64> {
        y_pred
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| 0.5 * (target - pred).powf(2.0))
            .collect()
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
        // final_predictions /= self.num_trees as f64;
        final_predictions
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array1, Array2}; // For floating-point assertions

    // #[test]
    // fn test_gradientboost_regressor() {
    //     let features = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 4.0, 5.0]).unwrap();
    //     let targets = Array1::from_vec(vec![1.1, 1.9, 3.9, 5.1]);
    //     let mut regressor = DecisionTreeRegressor::new(3, 1);

    //     if features.nrows() != targets.len() {
    //         panic!("Feature and target count mismatch");
    //     }

    //     regressor.fit(&features, &targets);
    //     let predictions = regressor.predict(&features);
    //     let rmse = calculate_rmse(&targets, &predictions);

    //     println!("RMSE: {:?}", rmse);
    //     assert_relative_eq!(rmse, 0.0, epsilon = 0.1);
    // }

    // fn calculate_rmse(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    //     let errors = actual - predicted;
    //     let mse = errors.mapv(|e| e.powi(2)).mean().unwrap();
    //     mse.sqrt()
    // }

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

        let mut gb_regressor = GBRegressor::new(50, 0.1, 4, 10, 3);
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


// fn softmax(residuals: &mut Vec<f64>) -> Result<()> {
//     todo!();
//     Ok(())
// }

// pub struct GBLearner<F, L>
// where F: Float,
//       L: Label,
// {
//   pub models: Vec<DecisionTree<F, L>>,
// }

// impl<F: Float, L: Label> GBLearner<F, L> {

// }

// // impl PredictInplace for GBLearner {
// //   fn predict_inplace<'a>(&'a self, x: &'a R, y: &mut T) {
// //       todo!()
// //   }

// //   fn default_target(&self, x: &R) -> T {
// //       todo!()
// //   }
// // }

// #[derive(Clone, Copy, Debug, PartialEq)]
// pub struct GBValidParams<F,L> {
//   n_estimators: usize,
//   learning_rate: f32,
//   d_tree_params: DecisionTreeParams<F, L>,
// }

// impl<F: Float, L: Label> GBValidParams<F, L> {
//   pub fn learning_rate(&self) -> f32 {
//       self.learning_rate
//   }

//   pub fn n_estimators(&self) -> usize {
//       self.n_estimators
//   }

//   pub fn d_tree_params(&self) -> DecisionTreeParams<F, L> {
//       self.d_tree_params.clone()
//   }
// }

// #[derive(Clone, Copy, Debug, PartialEq)]
// pub struct GBParams<F, L>(GBValidParams<F, L>);

// impl<F: Float, L: Label> GBParams<F, L> {
//   pub fn new() -> Self {
//     Self(GBValidParams {
//         learning_rate: 0.1,
//         n_estimators: 100,
//         d_tree_params: DecisionTreeParams::new(),
//     })
//   }

//   /// Sets the limit to how many stumps will be created
//   pub fn n_estimators(mut self, n_estimators: usize) -> Self {
//       self.0.n_estimators = n_estimators;
//       self
//   }

//   /// Sets the learning rate
//   pub fn learning_rate(mut self, learning_rate: f32) -> Self {
//       self.0.learning_rate = learning_rate;
//       self
//   }

//   /// Sets the params for the weak learner used in Gradient Boost
//   pub fn d_tree_params(mut self, d_tree_params: DecisionTreeParams<F, L>) -> Self {
//       self.0.d_tree_params = d_tree_params;
//       self
//   }
// }

// impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
//     for GBValidParams<F, L>
// where
//     D: Data<Elem = F>,
//     T: AsSingleTargets<Elem = L> + Labels<Elem = L> + std::fmt::Debug,
// {
//     type Object = GBLearner<F, L>;

//     fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {

//         let (x, y) = (dataset.records(), dataset.targets());

//         // let total = y.as_targets()
//         println!("{:?}", y);

//         // Trees vector initialization to store all the trees
//         let mut models = Vec::new();

//         // Initialize the classes an integer encoding
//         let targets: Vec<&L> = y.as_targets().iter().collect();

//         let mut target_nums: Vec<f64>;
//         for a_t in targets {
//           a_t.push(a_t as f64);
//         }

//         let target_sum: f64 = targets.iter().sum();
//         // let s = target.iter().sum();

//         let num_classes = classes.len();
//         let class_encoding: Vec<usize> = (0..num_classes).collect();

//         for a_target in dataset.as_targets() {

//         }
//         let mut residuals: Vec<f64> = Vec::new();

//         softmax(&mut residuals, data);

//         todo!()
//     }
// }
