use linfa::dataset::{AsSingleTargets, DatasetBase, Labels};
use linfa::traits::{Fit, FitWith, PredictInplace};
use linfa::{Float, Label};
use ndarray::{Array1, ArrayBase, ArrayView2, Data, Ix2};
use std::collections::HashMap;
use std::hash::Hash;

use crate::base_nb::{NaiveBayes, NaiveBayesValidParams};
use crate::error::{NaiveBayesError, Result};
use crate::hyperparams::{MultinomialNbParams, MultinomialNbValidParams};
use crate::{filter, ClassHistogram};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

impl<'a, F, L, D, T> NaiveBayesValidParams<'a, F, L, D, T> for MultinomialNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
}

impl<F, L, D, T> Fit<ArrayBase<D, Ix2>, T, NaiveBayesError> for MultinomialNbValidParams<F, L>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = MultinomialNb<F, L>;
    // Thin wrapper around the corresponding method of NaiveBayesValidParams
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        NaiveBayesValidParams::fit(self, dataset, None)
    }
}

impl<'a, F, L, D, T> FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError>
    for MultinomialNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type ObjectIn = Option<MultinomialNb<F, L>>;
    type ObjectOut = MultinomialNb<F, L>;

    fn fit_with(
        &self,
        model_in: Self::ObjectIn,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::ObjectOut> {
        let x = dataset.records();
        let y = dataset.as_single_targets();

        let mut model = match model_in {
            Some(temp) => temp,
            None => MultinomialNb {
                class_info: HashMap::new(),
            },
        };

        let yunique = dataset.labels();

        for class in yunique {
            // filter dataset for current class
            let xclass = filter(x.view(), y.view(), &class);

            // compute feature log probabilities and counts
            model
                .class_info
                .entry(class.clone())
                .or_insert_with(ClassHistogram::default)
                .update_with_smoothing(xclass.view(), self.alpha(), false);

            dbg!(&model.class_info.get(&class));
        }

        // update priors
        let class_count_sum = model
            .class_info
            .values()
            .map(|x| x.class_count)
            .sum::<usize>();

        for info in model.class_info.values_mut() {
            info.prior = F::cast(info.class_count) / F::cast(class_count_sum);
        }

        Ok(model)
    }
}

impl<F: Float, L: Label, D> PredictInplace<ArrayBase<D, Ix2>, Array1<L>> for MultinomialNb<F, L>
where
    D: Data<Elem = F>,
{
    // Thin wrapper around the corresponding method of NaiveBayes
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        NaiveBayes::predict_inplace(self, x, y);
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

/// Fitted Multinomial Naive Bayes classifier.
///
/// See [MultinomialNbParams] for more information on the hyper-parameters.
///
/// # Model assumptions
///
/// The family of Naive Bayes classifiers assume independence between variables. They do not model
/// moments between variables and lack therefore in modelling capability. The advantage is a linear
/// fitting time with maximum-likelihood training in a closed form.
///
/// # Model usage example
///
/// The example below creates a set of hyperparameters, and then uses it to fit a Multinomial Naive
/// Bayes classifier on provided data.
///
/// ```rust
/// use linfa_bayes::{MultinomialNbParams, MultinomialNbValidParams, Result};
/// use linfa::prelude::*;
/// use ndarray::array;
///
/// let x = array![
///     [-2., -1.],
///     [-1., -1.],
///     [-1., -2.],
///     [1., 1.],
///     [1., 2.],
///     [2., 1.]
/// ];
/// let y = array![1, 1, 1, 2, 2, 2];
/// let ds = DatasetView::new(x.view(), y.view());
///
/// // create a new parameter set with smoothing parameter equals `1`
/// let unchecked_params = MultinomialNbParams::new()
///     .alpha(1.0);
///
/// // fit model with unchecked parameter set
/// let model = unchecked_params.fit(&ds)?;
///
/// // transform into a verified parameter set
/// let checked_params = unchecked_params.check()?;
///
/// // update model with the verified parameters, this only returns
/// // errors originating from the fitting process
/// let model = checked_params.fit_with(Some(model), &ds)?;
/// # Result::Ok(())
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct MultinomialNb<F: PartialEq, L: Eq + Hash> {
    class_info: HashMap<L, ClassHistogram<F>>,
}

impl<F: Float, L: Label> MultinomialNb<F, L> {
    /// Construct a new set of hyperparameters
    pub fn params() -> MultinomialNbParams<F, L> {
        MultinomialNbParams::new()
    }
}

impl<F, L> NaiveBayes<'_, F, L> for MultinomialNb<F, L>
where
    F: Float,
    L: Label + Ord,
{
    // Compute unnormalized posterior log probability
    fn joint_log_likelihood(&self, x: ArrayView2<F>) -> HashMap<&L, Array1<F>> {
        let mut joint_log_likelihood = HashMap::new();
        for (class, info) in self.class_info.iter() {
            // Combine feature log probabilities and class priors to get log-likelihood for each class
            let jointi = info.prior.ln();
            let nij = x.dot(&info.feature_log_prob);
            joint_log_likelihood.insert(class, nij + jointi);
        }

        joint_log_likelihood
    }
}

#[cfg(test)]
mod tests {
    use super::{MultinomialNb, NaiveBayes, Result};
    use linfa::{
        traits::{Fit, FitWith, Predict},
        Dataset, DatasetView, Error,
    };

    use crate::{MultinomialNbParams, MultinomialNbValidParams};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Axis};
    use std::collections::HashMap;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<MultinomialNb<f64, usize>>();
        has_autotraits::<MultinomialNbValidParams<f64, usize>>();
        has_autotraits::<MultinomialNbParams<f64, usize>>();
    }

    #[test]
    fn test_multinomial_nb() -> Result<()> {
        let ds = Dataset::new(
            array![[1., 0.], [2., 0.], [3., 0.], [0., 1.], [0., 2.], [0., 3.]],
            array![1, 1, 1, 2, 2, 2],
        );

        let fitted_clf = MultinomialNb::params().fit(&ds)?;
        let pred = fitted_clf.predict(ds.records());

        assert_abs_diff_eq!(pred, ds.targets());

        let jll = fitted_clf.joint_log_likelihood(ds.records().view());
        let mut expected = HashMap::new();
        // Computed with sklearn.naive_bayes.MultinomialNB
        expected.insert(
            &1usize,
            array![
                -0.82667857,
                -0.96020997,
                -1.09374136,
                -2.77258872,
                -4.85203026,
                -6.93147181
            ],
        );

        expected.insert(
            &2usize,
            array![
                -2.77258872,
                -4.85203026,
                -6.93147181,
                -0.82667857,
                -0.96020997,
                -1.09374136
            ],
        );

        for (key, value) in jll.iter() {
            assert_abs_diff_eq!(value, expected.get(key).unwrap(), epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_mnb_fit_with() -> Result<()> {
        let x = array![[1., 0.], [2., 0.], [3., 0.], [0., 1.], [0., 2.], [0., 3.]];
        let y = array![1, 1, 1, 2, 2, 2];

        let clf = MultinomialNb::params();

        let model = x
            .axis_chunks_iter(Axis(0), 2)
            .zip(y.axis_chunks_iter(Axis(0), 2))
            .map(|(a, b)| DatasetView::new(a, b))
            .try_fold(None, |current, d| clf.fit_with(current, &d).map(Some))?
            .ok_or(Error::NotEnoughSamples)?;

        let pred = model.predict(&x);

        assert_abs_diff_eq!(pred, y);

        let jll = model.joint_log_likelihood(x.view());

        let mut expected = HashMap::new();
        // Computed with sklearn.naive_bayes.MultinomialNB
        expected.insert(
            &1usize,
            array![
                -0.82667857,
                -0.96020997,
                -1.09374136,
                -2.77258872,
                -4.85203026,
                -6.93147181
            ],
        );

        expected.insert(
            &2usize,
            array![
                -2.77258872,
                -4.85203026,
                -6.93147181,
                -0.82667857,
                -0.96020997,
                -1.09374136
            ],
        );

        for (key, value) in jll.iter() {
            assert_abs_diff_eq!(value, expected.get(key).unwrap(), epsilon = 1e-6);
        }

        Ok(())
    }
}
