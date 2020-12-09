//! Gaussian Naive Bayes (GaussianNB)
//!
//! Implements Gaussian Naive Bayes algorithm for classification. The likelihood
//! of the feature P(x_i | y) is assumed to be Gaussian, the mean and variance will
//! be estimated using maximum likelihood.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_stats::QuantileExt;

use crate::error::{BayesError, Result};
use linfa::dataset::{Dataset, Labels};
use linfa::traits::{Fit, Predict};
use linfa::Float;

/// Gaussian Naive Bayes (GaussianNB)
#[derive(Debug)]
pub struct GaussianNbParams<A> {
    // Prior probabilities of the classes
    priors: Option<Array1<A>>,
    // Boolean required for incrementally updating priors when not provided
    priors_provided: bool,
    // Required for calculation stability
    var_smoothing: f64,
}

impl<A> Default for GaussianNbParams<A> {
    fn default() -> Self {
        Self::params()
    }
}

impl<A> GaussianNbParams<A> {
    /// Create new GaussianNB model with default values for its parameters
    pub fn params() -> Self {
        GaussianNbParams {
            priors: None,
            priors_provided: false,
            var_smoothing: 1e-9,
            //class_count: None,
        }
    }

    // Prior probability of the classes, If set the priors will not be adjusted
    // according to data
    pub fn priors(mut self, priors: Array1<A>) -> Self {
        self.priors = Some(priors);
        self.priors_provided = true;
        self
    }

    // Specifies the portion of the largest variance of all the features that
    // is added to the variance for calculation stability
    pub fn var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }
}

impl<'a, A, L> Fit<'a, ArrayView2<'_, A>, L> for GaussianNbParams<A>
where
    A: Float + PartialEq + PartialOrd,
    L: Labels<Elem = usize>,
{
    type Object = Result<GaussianNb<A>>;

    /// Fit the model
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ndarray::array;
    /// # use linfa::Dataset;
    /// # use linfa_bayes::GaussianNbParams;
    /// # use linfa::traits::{Fit, Predict};
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let x = array![
    ///     [-2., -1.],
    ///     [-1., -1.],
    ///     [-1., -2.],
    ///     [1., 1.],
    ///     [1., 2.],
    ///     [2., 1.]
    /// ];
    /// let y = vec![1, 1, 1, 2, 2, 2];
    ///
    /// let data = Dataset::new(x.view(), &y);
    /// let model = GaussianNbParams::params().fit(&data)?;
    /// let pred = model.predict(x.view())?;
    ///
    /// assert_eq!(pred.to_vec(), y);
    /// # Ok(())
    /// # }
    /// ```
    fn fit(&self, dataset: &'a Dataset<ArrayView2<A>, L>) -> Self::Object {
        // We extract the unique classes in sorted order
        let mut unique_classes = dataset.targets.labels();
        unique_classes.sort_unstable();

        let mut model = GaussianNb::unfitted();
        if self.priors_provided {
            model.priors = self.priors.clone();
        }

        // We train the model
        model = self.fit_with(
            model,
            dataset.records.view(),
            &dataset.targets,
            &unique_classes,
        )?;

        Ok(model)
    }
}

impl<A: Float> GaussianNbParams<A> {
    /// Incrementally fit on a batch of samples
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ndarray::{array, Axis};
    /// # use linfa_bayes::{GaussianNb, GaussianNbParams};
    /// # use linfa::traits::Predict;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let x = array![
    ///     [-2., -1.],
    ///     [-1., -1.],
    ///     [-1., -2.],
    ///     [1., 1.],
    ///     [1., 2.],
    ///     [2., 1.]
    /// ];
    /// let y = array![1, 1, 1, 2, 2, 2];
    /// let classes = &[1, 2];
    ///
    /// let mut clf = GaussianNbParams::params();
    /// let mut model = GaussianNb::unfitted();
    ///
    /// for (x, y) in x
    ///     .axis_chunks_iter(Axis(0), 2)
    ///     .zip(y.axis_chunks_iter(Axis(0), 2))
    /// {
    ///     model = clf.fit_with(model, x, y.to_vec(), classes)?;
    /// }
    ///
    /// let pred = model.predict(x.view())?;
    ///
    /// assert_eq!(pred, y);
    /// # Ok(())
    /// # }
    /// ```
    pub fn fit_with<L: Labels<Elem = usize>>(
        &self,
        mut model: GaussianNb<A>,
        x: ArrayView2<A>,
        y: L,
        classes: &[usize],
    ) -> Result<GaussianNb<A>> {
        // If the ratio of the variance between dimensions is too small, it will cause
        // numerical errors. We address this by artificially boosting the variance
        // by `epsilon` (a small fraction of the variance of the largest feature)
        let epsilon = A::from(self.var_smoothing).unwrap()
            * *x.var_axis(Axis(0), A::from(0.).unwrap()).max()?;

        // If-Else conditional to determine if we are calling `fit_with`
        // for the first time
        // `If` branch signifies the function is being called for the first time
        if model.classes.is_none() {
            // We initialize `classes` with sorted unique categories found in `y`
            model.classes = Some(classes.to_owned());

            let nfeatures = x.ncols();

            // unwrap is safe since assigning `self.classes` is the first step
            let nclasses = model.classes.as_ref().unwrap().len();

            // We initialize mean, variance and class counts
            model.theta = Some(Array2::zeros((nclasses, nfeatures)));
            model.sigma = Some(Array2::zeros((nclasses, nfeatures)));
            model.class_count = Some(Array1::zeros(nclasses));

            // If priors is provided by the user, we perform checks to make
            // sure it is correct
            if let Some(ref priors) = model.priors {
                if priors.len() != nclasses {
                    return Err(BayesError::Priors(format!(
                        "The number of priors: ({}), does not match the number of classes: ({})",
                        priors.len(),
                        nclasses
                    )));
                }

                if (priors.sum() - A::from(1.0).unwrap()).abs() > A::from(1e-6).unwrap() {
                    return Err(BayesError::Priors(format!(
                        "The sum of priors: ({}), does not equal 1",
                        priors.sum()
                    )));
                }

                if priors.iter().any(|x| x < &A::from(0.).unwrap()) {
                    return Err(BayesError::Priors(
                        "Class priors cannot have negative values".to_string(),
                    ));
                }
            }
        } else {
            if x.ncols() != model.theta.as_ref().unwrap().ncols() {
                return Err(BayesError::Input(format!(
                    "Number of input columns: ({}), does not match the previous input: ({})",
                    x.ncols(),
                    model.theta.as_ref().unwrap().ncols()
                )));
            }

            // unwrap is safe because `self.sigma` is bound to have a value
            // if we are not calling `fit_with` for the first time
            model.sigma.as_mut().unwrap().mapv_inplace(|x| x - epsilon);
        }

        let yunique = y.labels();

        // We make sure there are no new classes in `y` that are not available
        // in `self.classes`
        let is_class_unavailable = yunique
            .iter()
            .any(|x| !model.classes.as_ref().unwrap().iter().any(|y| y == x));
        if is_class_unavailable {
            return Err(BayesError::Input(
                "Target labels in y, does not exist in the initial classes".to_string(),
            ));
        }

        for class in yunique.iter() {
            // unwrap is safe because we have made sure all elements of `yunique`
            // is in `self.classes`
            let position = model
                .classes
                .as_ref()
                .unwrap()
                .iter()
                .position(|y| y == class)
                .unwrap();

            // We filter x for records that correspond to the current class
            let xclass = Self::filter(&x, y.as_slice(), *class);

            // We count the number of occurances of the class
            let nclass = xclass.nrows();

            // We compute the update of the gaussian mean and variance
            let (theta_new, sigma_new) = Self::update_mean_variance(
                model.class_count.as_ref().unwrap()[position],
                &model.theta.as_ref().unwrap().slice(s![position, ..]),
                &model.sigma.as_ref().unwrap().slice(s![position, ..]),
                &xclass,
            );

            // We now update the mean, variance and class count
            model
                .theta
                .as_mut()
                .unwrap()
                .row_mut(position)
                .assign(&theta_new);
            model
                .sigma
                .as_mut()
                .unwrap()
                .row_mut(position)
                .assign(&sigma_new);
            let element = model
                .class_count
                .as_mut()
                .unwrap()
                .get_mut(position)
                .unwrap();
            *element += nclass;
        }

        // We add back the epsilon previously subtracted for numerical
        // calculation stability
        model.sigma.as_mut().unwrap().mapv_inplace(|x| x + epsilon);

        // If the priors are not provided by the user, we have to update
        // based on the current batch of data
        if !self.priors_provided {
            let class_count_sum = model.class_count.as_ref().unwrap().sum();
            let temp = model
                .class_count
                .as_ref()
                .unwrap()
                .mapv(|x| A::from(x).unwrap() / A::from(class_count_sum).unwrap());
            model.priors = Some(temp);
        }

        Ok(model)
    }

    // Compute online update of gaussian mean and variance
    fn update_mean_variance(
        count_old: usize,
        mu_old: &ArrayView1<A>,
        var_old: &ArrayView1<A>,
        x_new: &Array2<A>,
    ) -> (Array1<A>, Array1<A>) {
        // If incoming data is empty no updates required
        if x_new.nrows() == 0 {
            return (mu_old.to_owned(), var_old.to_owned());
        }

        let count_new = x_new.nrows();
        let mu_new = x_new.mean_axis(Axis(0)).unwrap();
        let var_new = x_new.var_axis(Axis(0), A::from(0.).unwrap());

        // If previous batch was empty, we send the new mean and variance calculated
        if count_old == 0 {
            return (mu_new, var_new);
        }

        let count_total = count_old + count_new;

        // Combine old and new mean, taking into consideration the number
        // of observations
        let mu_new_weighted = &mu_new * A::from(count_new).unwrap();
        let mu_old_weighted = mu_old * A::from(count_old).unwrap();
        let mu_weighted =
            (mu_new_weighted + mu_old_weighted).mapv(|x| x / A::from(count_total).unwrap());

        // Combine old and new variance, taking into consideration the number
        // of observations. this is achieved by combining the sum of squared
        // differences
        let ssd_old = var_old * A::from(count_old).unwrap();
        let ssd_new = var_new * A::from(count_new).unwrap();
        let weight = A::from(count_new * count_old).unwrap() / A::from(count_total).unwrap();
        let ssd_weighted = ssd_old + ssd_new + (mu_old - &mu_new).mapv(|x| weight * x.powi(2));
        let var_weighted = ssd_weighted.mapv(|x| x / A::from(count_total).unwrap());

        (mu_weighted, var_weighted)
    }

    // Returns a subset of x corresponding to the class specified by `ycondition`
    fn filter(x: &ArrayView2<A>, y: &[usize], ycondition: usize) -> Array2<A> {
        // We identify the row numbers corresponding to the class we are interested in
        let index = y
            .iter()
            .enumerate()
            .filter_map(|(i, y)| {
                if ycondition == *y {
                    return Some(i);
                }
                None
            })
            .collect::<Vec<_>>();

        // We subset x to only records corresponding to the class represented in `ycondition`
        let mut xsubset = Array2::zeros((index.len(), x.ncols()));
        index
            .iter()
            .enumerate()
            .for_each(|(i, &r)| xsubset.row_mut(i).assign(&x.slice(s![r, ..])));

        xsubset
    }
}

/// Fitted GaussianNB for predicting classes
#[derive(Debug)]
pub struct GaussianNb<A> {
    classes: Option<Vec<usize>>,
    class_count: Option<Array1<usize>>,
    priors: Option<Array1<A>>,
    theta: Option<Array2<A>>,
    sigma: Option<Array2<A>>,
}

impl<A> GaussianNb<A> {
    pub fn unfitted() -> Self {
        GaussianNb {
            classes: None,
            class_count: None,
            priors: None,
            theta: None,
            sigma: None,
        }
    }
}

impl<A: Float> Predict<ArrayView2<'_, A>, Result<Array1<usize>>> for GaussianNb<A> {
    /// Perform classification on incoming array
    fn predict(&self, x: ArrayView2<'_, A>) -> Result<Array1<usize>> {
        if self.classes.is_none() {
            return Err(BayesError::UntrainedModel(
                "Attempt to use an untrained model".to_string(),
            ));
        }
        let joint_log_likelihood = self.joint_log_likelihood(x);

        // Identify the class with the maximum log likelihood
        let output = joint_log_likelihood.map_axis(Axis(1), |x| {
            let i = x.argmax().unwrap();
            *self.classes.as_ref().unwrap().get(i).unwrap()
        });

        Ok(output)
    }
}

impl<A: Float> GaussianNb<A> {
    // Compute unnormalized posterior log probability
    fn joint_log_likelihood(&self, x: ArrayView2<A>) -> Array2<A> {
        let mut joint_log_likelihood =
            Array2::zeros((x.nrows(), self.classes.as_ref().unwrap().len()));

        for i in 0..self.classes.as_ref().unwrap().len() {
            let jointi = self.priors.as_ref().unwrap().get(i).unwrap().ln();

            let mut nij = self
                .sigma
                .as_ref()
                .unwrap()
                .row(i)
                .mapv(|x| A::from(2. * std::f64::consts::PI).unwrap() * x)
                .mapv(|x| x.ln())
                .sum();
            nij = A::from(-0.5).unwrap() * nij;

            let nij = ((x.to_owned() - self.theta.as_ref().unwrap().row(i)).mapv(|x| x.powi(2))
                / self.sigma.as_ref().unwrap().row(i))
            .sum_axis(Axis(1))
            .mapv(|x| x * A::from(0.5).unwrap())
            .mapv(|x| nij - x);

            joint_log_likelihood.column_mut(i).assign(&(nij + jointi));
        }

        joint_log_likelihood
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::Dataset;
    use ndarray::array;

    #[test]
    fn test_gaussian_nb() {
        let x = array![
            [-2., -1.],
            [-1., -1.],
            [-1., -2.],
            [1., 1.],
            [1., 2.],
            [2., 1.]
        ];
        let y = array![1, 1, 1, 2, 2, 2];

        let clf = GaussianNbParams::params();
        let data = Dataset::new(x.view(), y.view());
        let fitted_clf = clf.fit(&data).unwrap();
        let pred = fitted_clf.predict(x.view()).unwrap();
        assert_eq!(pred, y);

        let jll = fitted_clf.joint_log_likelihood(x.view());
        let expected = array![
            [-2.276946847943017, -38.27694652394301],
            [-1.5269468546930165, -25.52694663869301],
            [-2.276946847943017, -38.27694652394301],
            [-25.52694663869301, -1.5269468546930165],
            [-38.27694652394301, -2.276946847943017],
            [-38.27694652394301, -2.276946847943017]
        ];
        assert_eq!(jll, expected);
    }

    #[test]
    fn test_gnb_fit_with() {
        let x = array![
            [-2., -1.],
            [-1., -1.],
            [-1., -2.],
            [1., 1.],
            [1., 2.],
            [2., 1.]
        ];
        let y = array![1, 1, 1, 2, 2, 2];
        let classes = vec![1, 2];

        let clf = GaussianNbParams::params();
        let mut model = GaussianNb::unfitted();

        for (x, y) in x
            .axis_chunks_iter(Axis(0), 2)
            .zip(y.axis_chunks_iter(Axis(0), 2))
        {
            model = clf.fit_with(model, x, y, &classes).unwrap();
        }

        let pred = model.predict(x.view()).unwrap();

        assert_eq!(pred, y);

        let jll = model.joint_log_likelihood(x.view());
        let expected = array![
            [-2.276946847943017, -38.27694652394301],
            [-1.5269468546930165, -25.52694663869301],
            [-2.276946847943017, -38.27694652394301],
            [-25.52694663869301, -1.5269468546930165],
            [-38.27694652394301, -2.276946847943017],
            [-38.27694652394301, -2.276946847943017]
        ];
        assert_abs_diff_eq!(jll, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_gnb_priors1() {
        let x = array![
            [-2., -1.],
            [-1., -1.],
            [-1., -2.],
            [1., 1.],
            [1., 2.],
            [2., 1.]
        ];
        let y = array![1, 1, 1, 2, 2, 2];

        let clf = GaussianNbParams::params();
        let data = Dataset::new(x.view(), y.view());
        let fitted_clf = clf.fit(&data).unwrap();

        let expected = array![0.5, 0.5];

        assert_eq!(fitted_clf.priors.unwrap(), expected);
    }

    #[test]
    fn test_gnb_priors2() {
        let x = array![
            [-2., -1.],
            [-1., -1.],
            [-1., -2.],
            [1., 1.],
            [1., 2.],
            [2., 1.]
        ];
        let y = array![1, 1, 1, 2, 2, 2];

        let priors = array![0.3, 0.7];

        let clf: GaussianNbParams<f64> = GaussianNbParams::params().priors(priors.clone());
        let data = Dataset::new(x.view(), y.view());
        let fitted_clf = clf.fit(&data).unwrap();

        assert_eq!(fitted_clf.priors.unwrap(), priors);
    }
}
