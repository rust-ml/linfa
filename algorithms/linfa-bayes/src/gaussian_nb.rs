use linfa::dataset::{AsSingleTargets, DatasetBase, Labels};
use linfa::traits::{Fit, FitWith, PredictInplace};
use linfa::{Float, Label};
use ndarray::{Array1, ArrayBase, ArrayView2, Axis, Data, Ix2};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;
use std::hash::Hash;

use crate::base_nb::{NaiveBayes, NaiveBayesValidParams};
use crate::error::{NaiveBayesError, Result};
use crate::filter;
use crate::hyperparams::{GaussianNbParams, GaussianNbValidParams};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

impl<'a, F, L, D, T> NaiveBayesValidParams<'a, F, L, D, T> for GaussianNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
}

impl<F, L, D, T> Fit<ArrayBase<D, Ix2>, T, NaiveBayesError> for GaussianNbValidParams<F, L>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = GaussianNb<F, L>;

    // Thin wrapper around the corresponding method of NaiveBayesValidParams
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        NaiveBayesValidParams::fit(self, dataset, None)
    }
}

impl<'a, F, L, D, T> FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError>
    for GaussianNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type ObjectIn = Option<GaussianNb<F, L>>;
    type ObjectOut = GaussianNb<F, L>;

    fn fit_with(
        &self,
        model_in: Self::ObjectIn,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::ObjectOut> {
        let x = dataset.records();
        let y = dataset.as_single_targets();

        // If the ratio of the variance between dimensions is too small, it will cause
        // numerical errors. We address this by artificially boosting the variance
        // by `epsilon` (a small fraction of the variance of the largest feature)
        let epsilon = self.var_smoothing() * *x.var_axis(Axis(0), F::zero()).max()?;

        let mut model = match model_in {
            Some(mut temp) => {
                temp.class_info
                    .values_mut()
                    .for_each(|x| x.sigma -= epsilon);
                temp
            }
            None => GaussianNb {
                class_info: HashMap::new(),
            },
        };

        let yunique = dataset.labels();

        for class in yunique {
            // We filter for records that correspond to the current class
            let xclass = filter(x.view(), y.view(), &class);

            // We count the number of occurences of the class
            let nclass = xclass.nrows();

            // We compute the update of the gaussian mean and variance
            let class_info = model
                .class_info
                .entry(class)
                .or_insert_with(GaussianClassInfo::default);

            let (theta_new, sigma_new) = Self::update_mean_variance(class_info, xclass.view());

            // We now update the mean, variance and class count
            class_info.theta = theta_new;
            class_info.sigma = sigma_new;
            class_info.class_count += nclass;
        }

        // We add back the epsilon previously subtracted for numerical
        // calculation stability
        model
            .class_info
            .values_mut()
            .for_each(|x| x.sigma += epsilon);

        // We update the priors
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

impl<F: Float, L: Label, D> PredictInplace<ArrayBase<D, Ix2>, Array1<L>> for GaussianNb<F, L>
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

impl<F, L> GaussianNbValidParams<F, L>
where
    F: Float,
{
    // Compute online update of gaussian mean and variance
    fn update_mean_variance(
        info_old: &GaussianClassInfo<F>,
        x_new: ArrayView2<F>,
    ) -> (Array1<F>, Array1<F>) {
        // Deconstruct old state
        let (count_old, mu_old, var_old) = (info_old.class_count, &info_old.theta, &info_old.sigma);

        // If incoming data is empty no updates required
        if x_new.nrows() == 0 {
            return (mu_old.to_owned(), var_old.to_owned());
        }

        let count_new = x_new.nrows();

        // unwrap is safe because None is returned only when number of records
        // along the specified axis is 0, we return early if we have 0 rows
        let mu_new = x_new.mean_axis(Axis(0)).unwrap();
        let var_new = x_new.var_axis(Axis(0), F::zero());

        // If previous batch was empty, we send the new mean and variance calculated
        if count_old == 0 {
            return (mu_new, var_new);
        }

        let count_total = count_old + count_new;

        // Combine old and new mean, taking into consideration the number
        // of observations
        let mu_new_weighted = &mu_new * F::cast(count_new);
        let mu_old_weighted = mu_old * F::cast(count_old);
        let mu_weighted = (mu_new_weighted + mu_old_weighted).mapv(|x| x / F::cast(count_total));

        // Combine old and new variance, taking into consideration the number
        // of observations. This is achieved by combining the sum of squared
        // differences
        let ssd_old = var_old * F::cast(count_old);
        let ssd_new = var_new * F::cast(count_new);
        let weight = F::cast(count_new * count_old) / F::cast(count_total);
        let ssd_weighted = ssd_old + ssd_new + (mu_old - mu_new).mapv(|x| weight * x.powi(2));
        let var_weighted = ssd_weighted.mapv(|x| x / F::cast(count_total));

        (mu_weighted, var_weighted)
    }
}

/// Fitted Gaussian Naive Bayes classifier.
///
/// See [GaussianNbParams] for more information on the hyper-parameters.
///
/// # Model assumptions
///
/// The family of Naive Bayes classifiers assume independence between variables. They do not model
/// moments between variables and lack therefore in modelling capability. The advantage is a linear
/// fitting time with maximum-likelihood training in a closed form.
///
/// # Model usage example
///
/// The example below creates a set of hyperparameters, and then uses it to fit a Gaussian Naive Bayes
/// classifier on provided data.
///
/// ```rust
/// use linfa_bayes::{GaussianNbParams, GaussianNbValidParams, Result};
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
/// // create a new parameter set with variance smoothing equals `1e-5`
/// let unchecked_params = GaussianNbParams::new()
///     .var_smoothing(1e-5);
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
pub struct GaussianNb<F: PartialEq, L: Eq + Hash> {
    class_info: HashMap<L, GaussianClassInfo<F>>,
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Default, Clone, PartialEq)]
struct GaussianClassInfo<F> {
    class_count: usize,
    prior: F,
    theta: Array1<F>,
    sigma: Array1<F>,
}

impl<F: Float, L: Label> GaussianNb<F, L> {
    /// Construct a new set of hyperparameters
    pub fn params() -> GaussianNbParams<F, L> {
        GaussianNbParams::new()
    }
}

impl<F, L> NaiveBayes<'_, F, L> for GaussianNb<F, L>
where
    F: Float,
    L: Label + Ord,
{
    // Compute unnormalized posterior log probability
    fn joint_log_likelihood(&self, x: ArrayView2<F>) -> HashMap<&L, Array1<F>> {
        let mut joint_log_likelihood = HashMap::new();

        for (class, info) in self.class_info.iter() {
            let jointi = info.prior.ln();

            let mut nij = info
                .sigma
                .mapv(|x| F::cast(2. * std::f64::consts::PI) * x)
                .mapv(|x| x.ln())
                .sum();
            nij = F::cast(-0.5) * nij;

            let nij = ((x.to_owned() - &info.theta).mapv(|x| x.powi(2)) / &info.sigma)
                .sum_axis(Axis(1))
                .mapv(|x| x * F::cast(0.5))
                .mapv(|x| nij - x);

            joint_log_likelihood.insert(class, nij + jointi);
        }

        joint_log_likelihood
    }
}

#[cfg(test)]
mod tests {
    use super::{GaussianNb, NaiveBayes, Result};
    use linfa::{
        traits::{Fit, FitWith, Predict},
        DatasetView, Error,
    };

    use crate::gaussian_nb::GaussianClassInfo;
    use crate::{GaussianNbParams, GaussianNbValidParams, NaiveBayesError};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Axis};
    use std::collections::HashMap;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<GaussianNb<f64, usize>>();
        has_autotraits::<GaussianClassInfo<f64>>();
        has_autotraits::<GaussianNbParams<f64, usize>>();
        has_autotraits::<GaussianNbValidParams<f64, usize>>();
        has_autotraits::<NaiveBayesError>();
    }

    #[test]
    fn test_gaussian_nb() -> Result<()> {
        let x = array![
            [-2., -1.],
            [-1., -1.],
            [-1., -2.],
            [1., 1.],
            [1., 2.],
            [2., 1.]
        ];
        let y = array![1, 1, 1, 2, 2, 2];

        let data = DatasetView::new(x.view(), y.view());
        let fitted_clf = GaussianNb::params().fit(&data)?;
        let pred = fitted_clf.predict(&x);

        assert_abs_diff_eq!(pred, y);

        let jll = fitted_clf.joint_log_likelihood(x.view());

        // expected values from GaussianNB scikit-learn 1.6.1
        let mut expected = HashMap::new();
        expected.insert(
            &1usize,
            array![
                -2.276946847943017,
                -1.5269468546930165,
                -2.276946847943017,
                -25.52694663869301,
                -38.27694652394301,
                -38.27694652394301
            ],
        );
        expected.insert(
            &2usize,
            array![
                -38.27694652394301,
                -25.52694663869301,
                -38.27694652394301,
                -1.5269468546930165,
                -2.276946847943017,
                -2.276946847943017
            ],
        );

        assert_eq!(jll, expected);

        let expected_proba = array![
            [1.00000000e+00, 2.31952358e-16],
            [1.00000000e+00, 3.77513536e-11],
            [1.00000000e+00, 2.31952358e-16],
            [3.77513536e-11, 1.00000000e+00],
            [2.31952358e-16, 1.00000000e+00],
            [2.31952358e-16, 1.00000000e+00]
        ];

        let (y_pred_proba, classes) = fitted_clf.predict_proba(x.view());
        assert_eq!(classes, vec![&1usize, &2]);
        assert_abs_diff_eq!(expected_proba, y_pred_proba, epsilon = 1e-10);

        let (y_pred_log_proba, classes) = fitted_clf.predict_log_proba(x.view());
        assert_eq!(classes, vec![&1usize, &2]);
        assert_abs_diff_eq!(
            y_pred_proba.mapv(f64::ln),
            y_pred_log_proba,
            epsilon = 1e-10
        );

        Ok(())
    }

    #[test]
    fn test_gnb_fit_with() -> Result<()> {
        let x = array![
            [-2., -1.],
            [-1., -1.],
            [-1., -2.],
            [1., 1.],
            [1., 2.],
            [2., 1.]
        ];
        let y = array![1, 1, 1, 2, 2, 2];

        let clf = GaussianNb::params();

        let model = x
            .axis_chunks_iter(Axis(0), 2)
            .zip(y.axis_chunks_iter(Axis(0), 2))
            .map(|(a, b)| DatasetView::new(a, b))
            .fold(Ok(None), |current, d| clf.fit_with(current?, &d).map(Some))?
            .ok_or(Error::NotEnoughSamples)?;

        let pred = model.predict(&x);

        assert_abs_diff_eq!(pred, y);

        let jll = model.joint_log_likelihood(x.view());

        let mut expected = HashMap::new();
        expected.insert(
            &1usize,
            array![
                -2.276946847943017,
                -1.5269468546930165,
                -2.276946847943017,
                -25.52694663869301,
                -38.27694652394301,
                -38.27694652394301
            ],
        );
        expected.insert(
            &2usize,
            array![
                -38.27694652394301,
                -25.52694663869301,
                -38.27694652394301,
                -1.5269468546930165,
                -2.276946847943017,
                -2.276946847943017
            ],
        );

        for (key, value) in jll.iter() {
            assert_abs_diff_eq!(value, expected.get(key).unwrap(), epsilon = 1e-6);
        }

        Ok(())
    }
}
