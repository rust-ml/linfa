use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix2, Zip};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

use crate::error::{NaiveBayesError, Result};
use linfa::dataset::{AsTargets, DatasetBase, Labels};
use linfa::traits::FitWith;
use linfa::{Float, Label};

// Trait computing predictions for fitted Naive Bayes models
pub trait NaiveBayes<'a, F, L>
where
    F: Float,
    L: Label + Ord,
{
    /// Compute the unnormalized posterior log probabilities.
    /// The result is returned as an HashMap indexing log probabilities for each samples (eg x rows) by classes
    /// (eg jll\[class\] -> (n_samples,) array)
    fn joint_log_likelihood(&self, x: ArrayView2<F>) -> HashMap<&L, Array1<F>>;

    #[doc(hidden)]
    fn predict_inplace<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let joint_log_likelihood = self.joint_log_likelihood(x.view());

        // We store the classes and likelihood info in an vec and matrix
        // respectively for easier identification of the dominant class for
        // each input
        let nclasses = joint_log_likelihood.keys().len();
        let n = x.nrows();
        let mut classes = Vec::with_capacity(nclasses);
        let mut likelihood = Array2::zeros((nclasses, n));
        joint_log_likelihood
            .iter()
            .enumerate()
            .for_each(|(i, (&key, value))| {
                classes.push(key.clone());
                likelihood.row_mut(i).assign(value);
            });

        // Identify the class with the maximum log likelihood
        *y = likelihood.map_axis(Axis(0), |x| {
            let i = x.argmax().unwrap();
            classes[i].clone()
        });
    }

    /// Compute log-probability estimates for each sample wrt classes.
    /// The columns corresponds to classes in sorted order returned as the second output.
    fn predict_log_proba(&self, x: ArrayView2<F>) -> (Array2<F>, Vec<&L>) {
        let log_likelihood = self.joint_log_likelihood(x);

        let mut classes = log_likelihood.keys().cloned().collect::<Vec<_>>();
        classes.sort();

        let n_samples = x.nrows();
        let n_classes = log_likelihood.len();
        let mut log_prob_mat = Array2::<F>::zeros((n_samples, n_classes));

        Zip::from(log_prob_mat.columns_mut())
            .and(&classes)
            .for_each(|mut jll, &class| jll.assign(log_likelihood.get(class).unwrap()));

        let log_prob_x = log_prob_mat
            .mapv(|x| x.exp())
            .sum_axis(Axis(1))
            .mapv(|x| x.ln())
            .into_shape((n_samples, 1))
            .unwrap();

        (log_prob_mat - log_prob_x, classes)
    }

    /// Compute probability estimates for each sample wrt classes.
    /// The columns corresponds to classes in sorted order returned as the second output.  
    fn predict_proba(&self, x: ArrayView2<F>) -> (Array2<F>, Vec<&L>) {
        let (log_prob_mat, classes) = self.predict_log_proba(x);

        (log_prob_mat.mapv(|v| v.exp()), classes)
    }
}

// Common functionality for hyper-parameter sets of Naive Bayes models ready for estimation
pub(crate) trait NaiveBayesValidParams<'a, F, L, D, T>:
    FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    fn fit(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, T>,
        model_none: Self::ObjectIn,
    ) -> Result<Self::ObjectOut> {
        // We extract the unique classes in sorted order
        let mut unique_classes = dataset.targets.labels();
        unique_classes.sort_unstable();

        self.fit_with(model_none, dataset)
    }
}
