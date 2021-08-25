use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

use crate::error::{NaiveBayesError, Result};
use crate::hyperparams::{GaussianNbParams, GaussianNbValidParams};
use linfa::dataset::{AsTargets, DatasetBase, Labels};
use linfa::traits::{Fit, FitWith, PredictInplace};
use linfa::{Float, Label};

impl<F, L, D, T> Fit<ArrayBase<D, Ix2>, T, NaiveBayesError> for GaussianNbValidParams<F, L>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = GaussianNb<F, L>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        // We extract the unique classes in sorted order
        let mut unique_classes = dataset.targets.labels();
        unique_classes.sort_unstable();

        let mut model: Option<GaussianNb<_, _>> = None;

        // We train the model
        model = self.fit_with(model, dataset)?;

        Ok(model.unwrap())
    }
}

impl<'a, F, L, D, T> FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError>
    for GaussianNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    type ObjectIn = Option<GaussianNb<F, L>>;
    type ObjectOut = Option<GaussianNb<F, L>>;

    fn fit_with(
        &self,
        model_in: Self::ObjectIn,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::ObjectOut> {
        let x = dataset.records();
        let y = dataset.try_single_target()?;

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

        //let yunique = y.labels();
        let yunique = dataset.labels();

        for class in yunique {
            // We filter for records that correspond to the current class
            let xclass = Self::filter(x.view(), y.view(), &class);

            // We count the number of occurences of the class
            let nclass = xclass.nrows();

            // We compute the update of the gaussian mean and variance
            let mut class_info = model
                .class_info
                .entry(class)
                .or_insert_with(ClassInfo::default);

            let (theta_new, sigma_new) = Self::update_mean_variance(&class_info, xclass.view());

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

        Ok(Some(model))
    }
}

impl<F: Float, L: Label> GaussianNbValidParams<F, L> {
    // Compute online update of gaussian mean and variance
    fn update_mean_variance(
        info_old: &ClassInfo<F>,
        x_new: ArrayView2<F>,
    ) -> (Array1<F>, Array1<F>) {
        // deconstruct old state
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
        // of observations. this is achieved by combining the sum of squared
        // differences
        let ssd_old = var_old * F::cast(count_old);
        let ssd_new = var_new * F::cast(count_new);
        let weight = F::cast(count_new * count_old) / F::cast(count_total);
        let ssd_weighted = ssd_old + ssd_new + (mu_old - mu_new).mapv(|x| weight * x.powi(2));
        let var_weighted = ssd_weighted.mapv(|x| x / F::cast(count_total));

        (mu_weighted, var_weighted)
    }

    // Returns a subset of x corresponding to the class specified by `ycondition`
    fn filter(x: ArrayView2<F>, y: ArrayView1<L>, ycondition: &L) -> Array2<F> {
        // We identify the row numbers corresponding to the class we are interested in
        let index = y
            .into_iter()
            .enumerate()
            .filter_map(|(i, y)| match *ycondition == *y {
                true => Some(i),
                false => None,
            })
            .collect::<Vec<_>>();

        // We subset x to only records corresponding to the class represented in `ycondition`
        let mut xsubset = Array2::zeros((index.len(), x.ncols()));
        index
            .into_iter()
            .enumerate()
            .for_each(|(i, r)| xsubset.row_mut(i).assign(&x.slice(s![r, ..])));

        xsubset
    }
}

/// Fitted Gaussian Naive Bayes classifier
///
/// See [GaussianNbParams] for more information on the hyper-parameters.
///
/// # Model assumptions
///
/// The family of naive bayes classifiers assume independence between variables. They do not model
/// moments between variables and lack therefore in modelling capability. The advantage is a linear
/// fitting time with maximum-likelihood training in a closed form.
///
/// # Model estimation
///
/// You can fit a single model from a dataset
///
/// ```rust, ignore
/// use linfa::traits::Fit;
/// let model = GaussianNb::params().fit(&ds)?;
/// ```
///
/// or incrementally update a model
///
/// ```rust, ignore
/// use linfa::traits::FitWith;
/// let clf = GaussianNb::params();
/// let model = datasets.iter()
///     .try_fold(None, |prev_model, &ds| clf.fit_with(prev_model, ds))?
///     .unwrap();
/// ```
///
/// After fitting the model, you can use the [`Predict`](linfa::traits::Predict) variants to
/// predict new targets.
///
#[derive(Debug, Clone)]
pub struct GaussianNb<F, L> {
    class_info: HashMap<L, ClassInfo<F>>,
}

#[derive(Debug, Default, Clone)]
struct ClassInfo<F> {
    class_count: usize,
    prior: F,
    theta: Array1<F>,
    sigma: Array1<F>,
}

impl<F: Float, L: Label, D> PredictInplace<ArrayBase<D, Ix2>, Array1<L>> for GaussianNb<F, L>
where
    D: Data<Elem = F>,
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
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

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

impl<F: Float, L: Label> GaussianNb<F, L> {
    /// Construct a new set of hyperparameters
    pub fn params() -> GaussianNbParams<F, L> {
        GaussianNbParams::new()
    }

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
    use super::{GaussianNb, Result};
    use linfa::{
        traits::{Fit, FitWith, Predict},
        DatasetView,
    };

    use approx::assert_abs_diff_eq;
    use ndarray::{array, Axis};
    use std::collections::HashMap;

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
            .fold(None, |current, d| clf.fit_with(current, &d).unwrap())
            .unwrap();

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
