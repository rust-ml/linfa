use linfa::dataset::{AsSingleTargets, DatasetBase, Labels};
use linfa::traits::{Fit, FitWith, PredictInplace};
use linfa::{Float, Label};
use ndarray::{Array1, ArrayBase, ArrayView2, Axis, Data, Ix2, OwnedRepr};
use std::collections::HashMap;
use std::hash::Hash;

use crate::base_nb::{filter, NaiveBayes, NaiveBayesValidParams};
use crate::error::{NaiveBayesError, Result};
use crate::hyperparams::{BernoulliNbParams, BernoulliNbValidParams};

impl<'a, F, L, D, T> NaiveBayesValidParams<'a, F, L, D, T> for BernoulliNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
}

impl<F, L, D, T> Fit<ArrayBase<D, Ix2>, T, NaiveBayesError> for BernoulliNbValidParams<F, L>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = BernoulliNb<F, L>;
    // Thin wrapper around the corresponding method of NaiveBayesValidParams
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let model = NaiveBayesValidParams::fit(self, dataset, None)?;
        Ok(model.unwrap())
    }
}

impl<'a, F, L, D, T> FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError>
    for BernoulliNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type ObjectIn = Option<BernoulliNb<F, L>>;
    type ObjectOut = Option<BernoulliNb<F, L>>;

    fn fit_with(
        &self,
        model_in: Self::ObjectIn,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::ObjectOut> {
        let x = dataset.records();
        let y = dataset.as_single_targets();

        let mut model = match model_in {
            Some(temp) => temp,
            None => BernoulliNb {
                class_info: HashMap::new(),
                binarize: self.binarize().to_owned(),
            },
        };

        // Binarize data if the threshold is set
        let xbin = model.binarize(&x);

        // Calculate feature log probabilities
        let yunique = dataset.labels();
        for class in yunique {
            // We filter for records that correspond to the current class
            let xclass = filter(xbin.view(), y.view(), &class);

            // We count the number of occurrences of the class
            let nclass = xclass.nrows();

            // We compute the feature log probabilities and feature counts on
            // the slice corresponding to the current class
            let mut class_info = model
                .class_info
                .entry(class)
                .or_insert_with(BernoulliClassInfo::default);
            let (feature_log_prob, feature_count) =
                self.update_feature_log_prob(class_info, xclass.view());
            // We now update the total counts of each feature, feature
            // log probabilities, and class count
            class_info.feature_log_prob = feature_log_prob;
            class_info.feature_count = feature_count;
            class_info.class_count += nclass;
        }

        // Update the priors
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

impl<F: Float, L: Label, D> PredictInplace<ArrayBase<D, Ix2>, Array1<L>> for BernoulliNb<F, L>
where
    D: Data<Elem = F>,
{
    // Thin wrapper around the corresponding method of NaiveBayes
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        // Binarize data if the threshold is set
        let xbin = self.binarize(&x);
        NaiveBayes::predict_inplace(self, &xbin, y);
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

impl<'a, F, L> BernoulliNbValidParams<F, L>
where
    F: Float,
{
    // Update log probabilities of features given class
    fn update_feature_log_prob(
        &self,
        info_old: &BernoulliClassInfo<F>,
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
        // Compute total count (smoothed)
        let count = F::cast(x_new.nrows()) + self.alpha() * F::cast(2);
        // Compute log probabilities of each feature
        let feature_log_prob = feature_count_smoothed.mapv(|x| x.ln() - F::cast(count).ln());
        (feature_log_prob.to_owned(), feature_count.to_owned())
    }
}

/// Fitted Bernoulli Naive Bayes classifier.
///
/// See [BernoulliNbParams] for more information on the hyper-parameters.
///
/// # Model assumptions
///
/// The family of Naive Bayes classifiers assume independence between variables. They do not model
/// moments between variables and lack therefore in modelling capability. The advantage is a linear
/// fitting time with maximum-likelihood training in a closed form.
///
/// # Model usage example
///
/// The example below creates a set of hyperparameters, and then uses it to fit
/// a Bernoulli Naive Bayes classifier on provided data.
///
/// ```rust
/// use linfa_bayes::{BernoulliNbParams, BernoulliNbValidParams, Result};
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
/// let unchecked_params = BernoulliNbParams::new()
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
#[derive(Debug, Clone, PartialEq)]
pub struct BernoulliNb<F: PartialEq, L: Eq + Hash> {
    class_info: HashMap<L, BernoulliClassInfo<F>>,
    binarize: Option<F>,
}

#[derive(Debug, Default, Clone, PartialEq)]
struct BernoulliClassInfo<F> {
    class_count: usize,
    prior: F,
    feature_count: Array1<F>,
    feature_log_prob: Array1<F>,
}

impl<F: Float, L: Label> BernoulliNb<F, L> {
    /// Construct a new set of hyperparameters
    pub fn params() -> BernoulliNbParams<F, L> {
        BernoulliNbParams::new()
    }

    // Binarize data if the threshold is set
    fn binarize<'a, D>(&'a self, x: &'a ArrayBase<D, Ix2>) -> ArrayBase<OwnedRepr<F>, Ix2>
    where
        D: Data<Elem = F>,
    {
        let thr = self.binarize.unwrap_or(F::zero());
        let xbin = x.map(|v| if v > &thr { F::one() } else { F::zero() });
        return if self.binarize.is_some() {
            xbin.to_owned()
        } else {
            x.to_owned()
        };
    }
}

impl<'a, F, L> NaiveBayes<'a, F, L> for BernoulliNb<F, L>
where
    F: Float,
    L: Label + Ord,
{
    // Compute unnormalized posterior log probability
    fn joint_log_likelihood(&self, x: ArrayView2<F>) -> HashMap<&L, Array1<F>> {
        let mut joint_log_likelihood = HashMap::new();
        for (class, info) in self.class_info.iter() {
            // Combine feature log probabilities, their negatives, and class priors to
            // get log-likelihood for each class
            let neg_prob = info.feature_log_prob.map(|lp| (F::one() - lp.exp()).ln());
            let feature_log_prob = &info.feature_log_prob - &neg_prob;
            let jll = x.dot(&feature_log_prob);
            joint_log_likelihood.insert(class, jll + info.prior.ln() + neg_prob.sum());
        }

        joint_log_likelihood
    }
}

#[cfg(test)]
mod tests {
    use super::{BernoulliNb, NaiveBayes, Result};
    use linfa::{
        traits::{Fit, Predict},
        DatasetView,
    };

    use crate::bernoulli_nb::BernoulliClassInfo;
    use crate::{BernoulliNbParams, BernoulliNbValidParams};
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::collections::HashMap;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<BernoulliNb<f64, usize>>();
        has_autotraits::<BernoulliClassInfo<f64>>();
        has_autotraits::<BernoulliNbValidParams<f64, usize>>();
        has_autotraits::<BernoulliNbParams<f64, usize>>();
    }

    #[test]
    fn test_bernoulli_nb() -> Result<()> {
        let x = array![[1., 0.], [0., 0.], [1., 1.], [0., 1.]];
        let y = array![1, 1, 2, 2];
        let data = DatasetView::new(x.view(), y.view());

        let params = BernoulliNb::params().binarize(None);
        let fitted_clf = params.fit(&data)?;
        assert!(&fitted_clf.binarize.is_none());

        let pred = fitted_clf.predict(&x);
        assert_abs_diff_eq!(pred, y);

        let jll = fitted_clf.joint_log_likelihood(x.view());
        let mut expected = HashMap::new();
        expected.insert(
            &1usize,
            (array![0.1875f64, 0.1875, 0.0625, 0.0625]).map(|v| v.ln()),
        );

        expected.insert(
            &2usize,
            (array![0.0625f64, 0.0625, 0.1875, 0.1875,]).map(|v| v.ln()),
        );

        for (key, value) in jll.iter() {
            assert_abs_diff_eq!(value, expected.get(key).unwrap(), epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_text_class() -> Result<()> {
        // From https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html#tab:nbtoy
        let train = array![
            // C, B, S, M, T, J
            [2., 1., 0., 0., 0., 0.0f64],
            [2., 0., 1., 0., 0., 0.],
            [1., 0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 1., 1.],
        ];
        let y = array![1, 1, 1, 2];
        let test = array![[3., 0., 0., 0., 1., 1.0f64]];

        let data = DatasetView::new(train.view(), y.view());
        let fitted_clf = BernoulliNb::params().fit(&data)?;
        let pred = fitted_clf.predict(&test);

        assert_abs_diff_eq!(pred, array![2]);

        // See: https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
        let jll = fitted_clf.joint_log_likelihood(fitted_clf.binarize(&test).view());
        assert_abs_diff_eq!(jll.get(&1).unwrap()[0].exp(), 0.005, epsilon = 1e-3);
        assert_abs_diff_eq!(jll.get(&2).unwrap()[0].exp(), 0.022, epsilon = 1e-3);

        Ok(())
    }
}
