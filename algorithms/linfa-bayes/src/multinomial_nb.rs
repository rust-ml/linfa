use linfa::dataset::{AsTargets, DatasetBase, Labels};
use linfa::traits::{Fit, FitWith, PredictInplace};
use linfa::{Float, Label};
use ndarray::{Array1, ArrayBase, ArrayView2, Axis, Data, Ix2};
use std::collections::HashMap;

use crate::base_nb::{filter, NaiveBayes, NaiveBayesValidParams};
use crate::error::{NaiveBayesError, Result};
use crate::hyperparams::{MultinomialNbParams, MultinomialNbValidParams};

impl<'a, F, L, D, T> NaiveBayesValidParams<'a, F, L, D, T> for MultinomialNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
}

impl<F, L, D, T> Fit<ArrayBase<D, Ix2>, T, NaiveBayesError> for MultinomialNbValidParams<F, L>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = MultinomialNb<F, L>;
    // Thin wrapper around the corresponding method of NaiveBayesValidParams
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let model = NaiveBayesValidParams::fit(self, dataset, None)?;
        Ok(model.unwrap())
    }
}

impl<'a, F, L, D, T> FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError>
    for MultinomialNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    type ObjectIn = Option<MultinomialNb<F, L>>;
    type ObjectOut = Option<MultinomialNb<F, L>>;

    fn fit_with(
        &self,
        model_in: Self::ObjectIn,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::ObjectOut> {
        let x = dataset.records();
        let y = dataset.try_single_target()?;

        let mut model = match model_in {
            Some(temp) => temp,
            None => MultinomialNb {
                class_info: HashMap::new(),
            },
        };

        let yunique = dataset.labels();

        for class in yunique {
            // We filter for records that correspond to the current class
            let xclass = filter(x.view(), y.view(), &class);
            // We count the number of occurences of the class
            let nclass = xclass.nrows();

            // We compute the feature log probabilities and feature counts on the slice corresponding to the current class
            let mut class_info = model
                .class_info
                .entry(class)
                .or_insert_with(MultinomialClassInfo::default);
            let (feature_log_prob, feature_count) =
                self.update_feature_log_prob(class_info, xclass.view());
            // We now update the total counts of each feature, feature log probabilities, and class count
            class_info.feature_log_prob = feature_log_prob;
            class_info.feature_count = feature_count;
            class_info.class_count += nclass;
        }

        // We update the priors
        let class_count_sum = model
            .class_info
            .values()
            .map(|x| x.class_count)
            .sum::<usize>();
        for info in model.class_info.values_mut() {
            info.prior = F::cast(info.class_count) / F::cast(class_count_sum);
        }
        Ok(Some(model))
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

impl<'a, F, L> MultinomialNbValidParams<F, L>
where
    F: Float,
{
    // Update log probabilities of features given class
    fn update_feature_log_prob(
        &self,
        info_old: &MultinomialClassInfo<F>,
        x_new: ArrayView2<F>,
    ) -> (Array1<F>, Array1<F>) {
        // Deconstruct old state
        let (count_old, feature_log_prob_old, feature_count_old) = (
            &info_old.class_count,
            &info_old.feature_log_prob,
            &info_old.feature_count,
        );

        // If incoming data is empty no updates required
        if x_new.nrows() == 0 {
            return (
                feature_log_prob_old.to_owned(),
                feature_count_old.to_owned(),
            );
        }

        let feature_count_new = x_new.sum_axis(Axis(0));

        // If previous batch was empty, we send the new feature count calculated
        let feature_count = if count_old > &0 {
            feature_count_old + feature_count_new
        } else {
            feature_count_new
        };
        // Apply smoothing to feature counts
        let feature_count_smoothed = feature_count.clone() + self.alpha();
        // Compute total count over all (smoothed) features
        let count = feature_count_smoothed.sum();
        // Compute log probabilities of each feature
        let feature_log_prob = feature_count_smoothed.mapv(|x| x.ln() - F::cast(count).ln());
        (feature_log_prob.to_owned(), feature_count.to_owned())
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
#[derive(Debug, Clone)]
pub struct MultinomialNb<F, L> {
    class_info: HashMap<L, MultinomialClassInfo<F>>,
}

#[derive(Debug, Default, Clone)]
struct MultinomialClassInfo<F> {
    class_count: usize,
    prior: F,
    feature_count: Array1<F>,
    feature_log_prob: Array1<F>,
}

impl<F: Float, L: Label> MultinomialNb<F, L> {
    /// Construct a new set of hyperparameters
    pub fn params() -> MultinomialNbParams<F, L> {
        MultinomialNbParams::new()
    }
}

impl<'a, F, L, D> NaiveBayes<'a, F, L, D> for MultinomialNb<F, L>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
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
        DatasetView,
    };

    use approx::assert_abs_diff_eq;
    use ndarray::{array, Axis};
    use std::collections::HashMap;

    #[test]
    fn test_multinomial_nb() -> Result<()> {
        let x = array![[1., 0.], [2., 0.], [3., 0.], [0., 1.], [0., 2.], [0., 3.]];
        let y = array![1, 1, 1, 2, 2, 2];

        let data = DatasetView::new(x.view(), y.view());
        let fitted_clf = MultinomialNb::params().fit(&data)?;
        let pred = fitted_clf.predict(&x);

        assert_abs_diff_eq!(pred, y);

        // TODO
        let jll =
            NaiveBayes::<_, _, ndarray::OwnedRepr<_>>::joint_log_likelihood(&fitted_clf, x.view());
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
            .fold(None, |current, d| clf.fit_with(current, &d).unwrap())
            .unwrap();

        let pred = model.predict(&x);

        assert_abs_diff_eq!(pred, y);

        // TODO
        let jll = NaiveBayes::<_, _, ndarray::OwnedRepr<_>>::joint_log_likelihood(&model, x.view());

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
