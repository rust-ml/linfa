use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

use crate::error::{NaiveBayesError, Result};
use linfa::dataset::{AsTargets, DatasetBase, Labels};
use linfa::traits::FitWith;
use linfa::{Float, Label};

// Trait computing predictions for fitted Naive Bayes models
pub(crate) trait NaiveBayes<'a, F, L>: Send + Sync + Unpin + Sized
where
    F: Float,
    L: Label + Ord,
{
    fn joint_log_likelihood(&self, x: ArrayView2<F>) -> HashMap<&L, Array1<F>>;

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
}

// Common functionality for hyper-parameter sets of Naive Bayes models ready for estimation
pub(crate) trait NaiveBayesValidParams<'a, F, L, D, T>:
    FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError> + Send + Sync + Unpin + Sized
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

// Returns a subset of x corresponding to the class specified by `ycondition`
pub fn filter<F: Float, L: Label + Ord>(
    x: ArrayView2<F>,
    y: ArrayView1<L>,
    ycondition: &L,
) -> Array2<F> {
    // We identify the row numbers corresponding to the class we are interested in
    let index = y
        .into_iter()
        .enumerate()
        .filter_map(|(i, y)| (*ycondition == *y).then(|| i))
        .collect::<Vec<_>>();

    // We subset x to only records corresponding to the class represented in `ycondition`
    let mut xsubset = Array2::zeros((index.len(), x.ncols()));
    index
        .into_iter()
        .enumerate()
        .for_each(|(i, r)| xsubset.row_mut(i).assign(&x.slice(s![r, ..])));

    xsubset
}
