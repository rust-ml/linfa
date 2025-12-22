use crate::AdaBoostValidParams;
use linfa::{
    dataset::{AsTargets, AsTargetsMut, FromTargetArrayOwned},
    error::Error,
    traits::*,
    DatasetBase,
};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::distributions::WeightedIndex;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::rand::Rng;
use std::{cmp::Eq, collections::HashMap, hash::Hash};

/// A fitted AdaBoost ensemble classifier.
///
/// ## Structure
///
/// AdaBoost (Adaptive Boosting) is an ensemble learning method that combines multiple weak learners
/// into a strong classifier. Unlike bagging methods (like Random Forest), AdaBoost trains learners
/// sequentially, where each new learner focuses more on examples that previous learners misclassified.
///
/// Each fitted model `M` has an associated weight (alpha) that represents its contribution to the
/// final prediction. Models that perform better on their training data receive higher weights.
///
/// ## Algorithm Overview
///
/// Given a [DatasetBase](DatasetBase) denoted as `D` with `n` samples:
/// 1. Initialize sample weights uniformly: `w_i = 1/n` for all samples
/// 2. For each iteration `t` from 1 to T (number of estimators):
///    a. Train base learner on weighted dataset
///    b. Calculate weighted error rate
///    c. Compute model weight (alpha) based on error
///    d. Update sample weights: increase weights for misclassified samples
///    e. Normalize sample weights
///
/// ## Prediction Algorithm
///
/// The final prediction is computed using weighted majority voting:
/// - Each model's prediction is weighted by its alpha value
/// - The class with the highest weighted vote is selected
///
/// ## Example
///
/// ```no_run
/// use linfa::prelude::{Fit, Predict};
/// use linfa_ensemble::AdaBoostParams;
/// use linfa_trees::DecisionTree;
/// use ndarray_rand::rand::SeedableRng;
/// use rand::rngs::SmallRng;
///
/// // Load Iris dataset
/// let mut rng = SmallRng::seed_from_u64(42);
/// let (train, test) = linfa_datasets::iris()
///     .shuffle(&mut rng)
///     .split_with_ratio(0.8);
///
/// // Train AdaBoost with decision tree stumps
/// let adaboost_model = AdaBoostParams::new(DecisionTree::params().max_depth(Some(1)))
///     .n_estimators(50)
///     .learning_rate(1.0)
///     .fit(&train)
///     .unwrap();
///
/// // Make predictions on the test set
/// let predictions = adaboost_model.predict(&test);
/// ```
///
/// ## References
///
/// * Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning
///   and an application to boosting. Journal of Computer and System Sciences, 55(1), 119-139.
/// * [Scikit-Learn AdaBoost Documentation](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
/// * [An Introduction to Statistical Learning](https://www.statlearning.com/), Chapter 8
#[derive(Debug, Clone)]
pub struct AdaBoost<M, L> {
    /// The fitted base learner models
    pub models: Vec<M>,
    /// The weight (alpha) for each model in the ensemble
    pub model_weights: Vec<f64>,
    /// The unique class labels seen during training
    pub classes: Vec<L>,
}

impl<M, L> AdaBoost<M, L> {
    /// Returns the number of estimators in the ensemble
    pub fn n_estimators(&self) -> usize {
        self.models.len()
    }

    /// Returns the model weights (alpha values)
    pub fn weights(&self) -> &[f64] {
        &self.model_weights
    }
}

impl<F: Clone, T, M, L> PredictInplace<Array2<F>, T> for AdaBoost<M, L>
where
    M: PredictInplace<Array2<F>, T>,
    <T as AsTargets>::Elem: Copy + Eq + Hash + std::fmt::Debug + Into<usize>,
    T: AsTargets + AsTargetsMut<Elem = <T as AsTargets>::Elem>,
    usize: Into<<T as AsTargets>::Elem>,
{
    fn predict_inplace(&self, x: &Array2<F>, y: &mut T) {
        let y_array = y.as_targets();
        assert_eq!(
            x.nrows(),
            y_array.len_of(Axis(0)),
            "The number of data points must match the number of outputs."
        );

        // Collect predictions from all models
        let mut all_predictions = Vec::with_capacity(self.models.len());
        for model in &self.models {
            let mut pred = model.default_target(x);
            model.predict_inplace(x, &mut pred);
            all_predictions.push(pred);
        }

        // Create a map for each sample to accumulate weighted votes
        let mut prediction_maps = y_array.map(|_| HashMap::new());

        // Accumulate weighted predictions from each model
        for (model_idx, prediction) in all_predictions.iter().enumerate() {
            let pred_array = prediction.as_targets();
            let weight = self.model_weights[model_idx];

            // For each sample, add the model's weighted prediction
            for (vote_map, &pred_val) in prediction_maps.iter_mut().zip(pred_array.iter()) {
                let class_idx: usize = pred_val.into();
                *vote_map.entry(class_idx).or_insert(0.0) += weight;
            }
        }

        // For each sample, select the class with the highest weighted vote
        let final_predictions = prediction_maps.map(|votes| {
            votes
                .iter()
                .max_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())
                .map(|(k, _)| (*k).into())
                .expect("No predictions found")
        });

        // Write final predictions to output
        let mut y_array_mut = y.as_targets_mut();
        for (y, pred) in y_array_mut.iter_mut().zip(final_predictions.iter()) {
            *y = *pred;
        }
    }

    fn default_target(&self, x: &Array2<F>) -> T {
        self.models[0].default_target(x)
    }
}

impl<D, T, P, R> Fit<Array2<D>, T, Error> for AdaBoostValidParams<P, R>
where
    D: Clone + ndarray::ScalarOperand,
    T: FromTargetArrayOwned<Owned = T> + AsTargets + Clone,
    T::Elem: Copy + Eq + Hash + std::fmt::Debug + Into<usize>,
    P: Fit<Array2<D>, T, Error> + Clone,
    P::Object: PredictInplace<Array2<D>, T>,
    R: Rng + Clone,
    usize: Into<T::Elem>,
{
    type Object = AdaBoost<P::Object, T::Elem>;

    fn fit(
        &self,
        dataset: &DatasetBase<Array2<D>, T>,
    ) -> core::result::Result<Self::Object, Error> {
        let n_samples = dataset.records.nrows();

        if n_samples == 0 {
            return Err(Error::Parameters(
                "Cannot fit AdaBoost on empty dataset".to_string(),
            ));
        }

        // Extract unique class labels from target array
        let target_array = dataset.targets.as_targets();
        let mut classes_set: Vec<T::Elem> = target_array
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        // Sort by converting to usize for ordering
        classes_set.sort_unstable_by_key(|x| (*x).into());

        if classes_set.len() < 2 {
            return Err(Error::Parameters(
                "AdaBoost requires at least 2 classes".to_string(),
            ));
        }

        // Initialize sample weights uniformly
        let mut sample_weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

        let mut models = Vec::with_capacity(self.n_estimators);
        let mut model_weights = Vec::with_capacity(self.n_estimators);

        let mut rng = self.rng.clone();

        for iteration in 0..self.n_estimators {
            // Normalize weights to sum to 1
            let weight_sum = sample_weights.sum();
            if weight_sum <= 0.0 {
                return Err(Error::NotConverged(format!(
                    "Sample weights sum to zero at iteration {}",
                    iteration
                )));
            }
            sample_weights /= weight_sum;

            // Resample dataset according to sample weights
            // This is the practical implementation of AdaBoost when base learners don't support weights
            let dist = WeightedIndex::new(sample_weights.iter().copied())
                .map_err(|_| Error::Parameters("Invalid sample weights".to_string()))?;

            let bootstrap_indices: Vec<usize> =
                (0..n_samples).map(|_| dist.sample(&mut rng)).collect();

            // Create bootstrap dataset by selecting rows according to weights
            let bootstrap_records = dataset.records.select(Axis(0), &bootstrap_indices);
            let bootstrap_targets_array = target_array.select(Axis(0), &bootstrap_indices);

            // Convert to owned target type using new_targets
            let bootstrap_targets = T::new_targets(bootstrap_targets_array);
            let bootstrap_dataset = DatasetBase::new(bootstrap_records, bootstrap_targets);

            // Fit base learner on resampled dataset
            let model = self.model_params.fit(&bootstrap_dataset).map_err(|e| {
                Error::NotConverged(format!(
                    "Base learner failed to fit at iteration {}: {}",
                    iteration, e
                ))
            })?;

            // Make predictions on training data
            let mut predictions = model.default_target(&dataset.records);
            model.predict_inplace(&dataset.records, &mut predictions);
            let pred_array = predictions.as_targets();

            // Calculate weighted error
            let mut weighted_error = 0.0;
            for ((true_label, pred_label), weight) in target_array
                .iter()
                .zip(pred_array.iter())
                .zip(sample_weights.iter())
            {
                let true_idx: usize = (*true_label).into();
                let pred_idx: usize = (*pred_label).into();

                if true_idx != pred_idx {
                    weighted_error += *weight;
                }
            }

            // Handle edge cases for weighted error
            if weighted_error <= 0.0 {
                // Perfect prediction - add model with maximum weight and stop
                model_weights.push(10.0); // Large weight for perfect classifier
                models.push(model);
                break;
            }

            // For multi-class SAMME, check if error rate is above the random guessing threshold
            let k = classes_set.len() as f64;
            let error_threshold = (k - 1.0) / k;

            if weighted_error >= error_threshold {
                // Worse than random guessing for multi-class - don't add this model
                if models.is_empty() {
                    return Err(Error::NotConverged(format!(
                        "First base learner performs worse than random guessing (error: {:.4}, threshold: {:.4})",
                        weighted_error, error_threshold
                    )));
                }
                break;
            }

            // Calculate model weight (alpha) using SAMME algorithm
            // For multi-class: alpha = learning_rate * (log((1 - error) / error) + log(K - 1))
            // where K is number of classes
            let error_ratio = (1.0 - weighted_error) / weighted_error;
            let alpha = self.learning_rate * (error_ratio.ln() + (k - 1.0).ln());

            // Update sample weights
            for ((true_label, pred_label), weight) in target_array
                .iter()
                .zip(pred_array.iter())
                .zip(sample_weights.iter_mut())
            {
                let true_idx: usize = (*true_label).into();
                let pred_idx: usize = (*pred_label).into();

                if true_idx != pred_idx {
                    // Increase weight for misclassified samples
                    *weight *= alpha.exp();
                }
            }

            model_weights.push(alpha);
            models.push(model);
        }

        if models.is_empty() {
            return Err(Error::NotConverged(
                "No models were successfully trained".to_string(),
            ));
        }

        Ok(AdaBoost {
            models,
            model_weights,
            classes: classes_set,
        })
    }
}
