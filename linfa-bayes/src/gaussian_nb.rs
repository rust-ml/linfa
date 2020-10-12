//! Gaussian Naive Bayes (GaussianNB)
//!
//! Implements Gaussian Naive Bayes algorithm for classification. The likelihood
//! of the feature P(x_i | y) is assumed to be Gaussian, the mean and variance will
//! be estimated using maximum likelihood.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_stats::QuantileExt;

use crate::error::{BayesError, Result};
use crate::Float;

/// Gaussian Naive Bayes (GaussianNB)
#[derive(Debug)]
pub struct GaussianNb<A> {
    // Prior probabilities of the classes
    priors: Option<Array1<A>>,
    // Boolean required for incrementally updating priors when not provided
    priors_provided: bool,
    // Required for calculation stability
    var_smoothing: f64,
    // Class labels known to the classifier
    classes: Option<Array1<A>>,
    // Mean of each feature per class
    theta: Option<Array2<A>>,
    // Variance of each feature per class
    sigma: Option<Array2<A>>,
    // Number of training samples observed in each class
    class_count: Option<Array1<A>>,
}

impl<A> Default for GaussianNb<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> GaussianNb<A> {
    /// Create new GaussianNB model with default values for its parameters
    pub fn new() -> Self {
        GaussianNb {
            priors: None,
            priors_provided: false,
            var_smoothing: 1e-9,
            classes: None,
            theta: None,
            sigma: None,
            class_count: None,
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

impl<A: Float + PartialEq + PartialOrd> GaussianNb<A> {
    /// Fit the model
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ndarray::array;
    /// # use linfa_bayes::GaussianNb;
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
    /// let y = array![1., 1., 1., 2., 2., 2.];
    ///
    /// let mut clf = GaussianNb::new();
    /// let fitted_clf = clf.fit(&x, &y)?;
    /// let pred = fitted_clf.predict(&x);
    ///
    /// assert_eq!(pred, y);
    /// # Ok(())
    /// # }
    /// ```
    pub fn fit(&mut self, x: &Array2<A>, y: &Array1<A>) -> Result<FittedGaussianNb<A>> {
        // We extract the unique classes in sorted order
        let unique_classes = GaussianNb::unique(&y.view());

        // We train the model
        self.partial_fit(&x.view(), &y.view(), &unique_classes)?;

        // We access the trained model
        let classifier = self.get_predictor()?;

        Ok(classifier)
    }

    /// Incrementally fit on a batch of samples
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ndarray::{array, Axis};
    /// # use linfa_bayes::GaussianNb;
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
    /// let y = array![1., 1., 1., 2., 2., 2.];
    /// let classes = array![1., 2.];
    ///
    /// let mut clf = GaussianNb::new();
    ///
    /// for (x, y) in x
    ///     .axis_chunks_iter(Axis(0), 2)
    ///     .zip(y.axis_chunks_iter(Axis(0), 2))
    /// {
    ///     clf.partial_fit(&x, &y, &classes)?;
    /// }
    ///
    /// let fitted_clf = clf.get_predictor()?;
    /// let pred = fitted_clf.predict(&x);
    ///
    /// assert_eq!(pred, y);
    /// # Ok(())
    /// # }
    /// ```
    pub fn partial_fit(
        &mut self,
        x: &ArrayView2<A>,
        y: &ArrayView1<A>,
        classes: &Array1<A>,
    ) -> Result<()> {
        // If the ratio of the variance between dimensions is too small, it will cause
        // numerical errors. We address this by artificially boosting the variance
        // by `epsilon` (a small fraction of the variance of the largest feature)
        let epsilon = A::from(self.var_smoothing).unwrap()
            * *x.var_axis(Axis(0), A::from(0.).unwrap()).max()?;

        // If-Else conditional to determine if we are calling `partial_fit`
        // for the first time
        // `If` branch signifies the function is being called for the first time
        if self.classes.is_none() {
            // We initialize `classes` with sorted unique categories found in `y`
            //self.classes = Some(Self::unique(&y));
            self.classes = Some(classes.to_owned());

            let nfeatures = x.ncols();

            // unwrap is safe since assigning `self.classes` is the first step
            let nclasses = self.classes.as_ref().unwrap().len();

            // We initialize mean, variance and class counts
            self.theta = Some(Array2::zeros((nclasses, nfeatures)));
            self.sigma = Some(Array2::zeros((nclasses, nfeatures)));
            self.class_count = Some(Array1::zeros(nclasses));

            // If priors is provided by the user, we perform checks to make
            // sure it is correct
            if let Some(ref priors) = self.priors {
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
            if x.ncols() != self.theta.as_ref().unwrap().ncols() {
                return Err(BayesError::Input(format!(
                    "Number of input columns: ({}), does not match the previous input: ({})",
                    x.ncols(),
                    self.theta.as_ref().unwrap().ncols()
                )));
            }

            // unwrap is safe because `self.sigma` is bound to have a value
            // if we are not calling `partial_fit` for the first time
            self.sigma.as_mut().unwrap().mapv_inplace(|x| x - epsilon);
        }

        let yunique = Self::unique(&y);

        // We make sure there are no new classes in `y` that are not available
        // in `self.classes`
        let is_class_unavailable = yunique
            .iter()
            .any(|x| !self.classes.as_ref().unwrap().iter().any(|y| y == x));
        if is_class_unavailable {
            return Err(BayesError::Input(
                "Target labels in y, does not exist in the initial classes".to_string(),
            ));
        }

        for class in yunique.iter() {
            // unwrap is safe because we have made sure all elements of `yunique`
            // is in `self.classes`
            let position = self
                .classes
                .as_ref()
                .unwrap()
                .iter()
                .position(|y| y == class)
                .unwrap();

            // We filter x for records that correspond to the current class
            let xclass = Self::filter(&x, &y, *class)?;

            // We count the number of occurances of the class
            let nclass = xclass.nrows();

            // We compute the update of the gaussian mean and variance
            let (theta_new, sigma_new) = Self::update_mean_variance(
                self.class_count.as_ref().unwrap()[position],
                &self.theta.as_ref().unwrap().slice(s![position, ..]),
                &self.sigma.as_ref().unwrap().slice(s![position, ..]),
                &xclass,
            );

            // We now update the mean, variance and class count
            self.theta
                .as_mut()
                .unwrap()
                .row_mut(position)
                .assign(&theta_new);
            self.sigma
                .as_mut()
                .unwrap()
                .row_mut(position)
                .assign(&sigma_new);
            let element = self
                .class_count
                .as_mut()
                .unwrap()
                .get_mut(position)
                .unwrap();
            *element += A::from(nclass).unwrap();
        }

        // We add back the epsilon previously subtracted for numerical
        // calculation stability
        self.sigma.as_mut().unwrap().mapv_inplace(|x| x + epsilon);

        // If the priors are not provided by the user, we have to update
        // based on the current batch of data
        if !self.priors_provided {
            let class_count_sum = self.class_count.as_ref().unwrap().sum();
            let temp = self
                .class_count
                .as_ref()
                .unwrap()
                .mapv(|x| x / class_count_sum);
            self.priors = Some(temp);
        }

        Ok(())
    }

    // Compute online update of gaussian mean and variance
    fn update_mean_variance(
        count_old: A,
        mu_old: &ArrayView1<A>,
        var_old: &ArrayView1<A>,
        x_new: &Array2<A>,
    ) -> (Array1<A>, Array1<A>) {
        // If incoming data is empty no updates required
        if x_new.nrows() == 0 {
            return (mu_old.to_owned(), var_old.to_owned());
        }

        let count_new = A::from(x_new.nrows()).unwrap();
        let mu_new = x_new.mean_axis(Axis(0)).unwrap();
        let var_new = x_new.var_axis(Axis(0), A::from(0.).unwrap());

        // If previous batch was empty, we send the new mean and variance calculated
        if count_old == A::from(0.).unwrap() {
            return (mu_new, var_new);
        }

        let count_total = count_old + count_new;

        // Combine old and new mean, taking into consideration the number
        // of observations
        let mu_new_weighted = &mu_new * A::from(count_new).unwrap();
        let mu_old_weighted = mu_old * A::from(count_old).unwrap();
        let mu_weighted = (mu_new_weighted + mu_old_weighted).mapv(|x| x / count_total);

        // Combine old and new variance, taking into consideration the number
        // of observations. this is achieved by combining the sum of squared
        // differences
        let ssd_old = var_old * A::from(count_old).unwrap();
        let ssd_new = var_new * A::from(count_new).unwrap();
        let weight = A::from(count_new * count_old).unwrap() / count_total;
        let ssd_weighted = ssd_old + ssd_new + (mu_old - &mu_new).mapv(|x| weight * x.powi(2));
        let var_weighted = ssd_weighted.mapv(|x| x / count_total);

        (mu_weighted, var_weighted)
    }

    /// Get the trained model, which can be used for prediction
    pub fn get_predictor(&self) -> Result<FittedGaussianNb<A>> {
        if self.classes.is_none() {
            return Err(BayesError::UntrainedModel(
                "Attempt to access untrained model".to_string(),
            ));
        }

        Ok(FittedGaussianNb {
            classes: self.classes.as_ref().unwrap().to_owned(),
            priors: self.priors.as_ref().unwrap().to_owned(),
            theta: self.theta.as_ref().unwrap().to_owned(),
            sigma: self.sigma.as_ref().unwrap().to_owned(),
        })
    }

    // Extract unique elements of the array in sorted order
    fn unique(y: &ArrayView1<A>) -> Array1<A> {
        // We are identifying unique classes in y,
        // ndarray doesn't provide methods for extracting unique elements,
        // So we are converting it to a Vec
        let mut unique_classes = y.to_vec();
        unique_classes.sort_by(|x, y| x.partial_cmp(y).unwrap());
        unique_classes.dedup();

        Array1::from(unique_classes)
    }

    // Returns a subset of x corresponding to the class specified by `ycondition`
    fn filter(x: &ArrayView2<A>, y: &ArrayView1<A>, ycondition: A) -> Result<Array2<A>> {
        // We identify the row numbers corresponding to the class we are interested in
        let index: Vec<_> = y
            .indexed_iter()
            .filter_map(|(i, y)| {
                if (ycondition - *y).abs() < A::from(1e-6).unwrap() {
                    return Some(i);
                }
                None
            })
            .collect();

        // We subset x to only records corresponding to the class represented in `ycondition`
        let mut xsubset = Array2::zeros((index.len(), x.ncols()));
        index
            .iter()
            .enumerate()
            .for_each(|(i, &r)| xsubset.row_mut(i).assign(&x.slice(s![r, ..])));

        Ok(xsubset)
    }
}

/// Fitted GaussianNB for predicting classes
#[derive(Debug)]
pub struct FittedGaussianNb<A> {
    classes: Array1<A>,
    priors: Array1<A>,
    theta: Array2<A>,
    sigma: Array2<A>,
}

impl<A: Float> FittedGaussianNb<A> {
    /// Perform classification on incoming array
    pub fn predict(&self, x: &Array2<A>) -> Array1<A> {
        let joint_log_likelihood = self.joint_log_likelihood(x);

        // Identify the class with the maximum log likelihood
        joint_log_likelihood.map_axis(Axis(1), |x| {
            let i = x.argmax().unwrap();
            *self.classes.get(i).unwrap()
        })
    }

    // Compute unnormalized posterior log probability
    fn joint_log_likelihood(&self, x: &Array2<A>) -> Array2<A> {
        let mut joint_log_likelihood = Array2::zeros((x.nrows(), self.classes.len()));

        for i in 0..self.classes.len() {
            let jointi = self.priors.get(i).unwrap().ln();

            let mut nij = self
                .sigma
                .row(i)
                .mapv(|x| A::from(2. * std::f64::consts::PI).unwrap() * x)
                .mapv(|x| x.ln())
                .sum();
            nij = A::from(-0.5).unwrap() * nij;

            let nij = ((x - &self.theta.row(i)).mapv(|x| x.powi(2)) / self.sigma.row(i))
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
        let y = array![1., 1., 1., 2., 2., 2.];

        let mut clf = GaussianNb::new();
        let fitted_clf = clf.fit(&x, &y).unwrap();
        let pred = fitted_clf.predict(&x);
        assert_eq!(pred, y);

        let jll = fitted_clf.joint_log_likelihood(&x);
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
    fn test_gnb_partial_fit() {
        let x = array![
            [-2., -1.],
            [-1., -1.],
            [-1., -2.],
            [1., 1.],
            [1., 2.],
            [2., 1.]
        ];
        let y = array![1., 1., 1., 2., 2., 2.];
        let classes = array![1., 2.];

        let mut clf = GaussianNb::new();

        for (x, y) in x
            .axis_chunks_iter(Axis(0), 2)
            .zip(y.axis_chunks_iter(Axis(0), 2))
        {
            clf.partial_fit(&x, &y, &classes).unwrap();
        }

        let fitted_clf = clf.get_predictor().unwrap();
        let pred = fitted_clf.predict(&x);

        assert_eq!(pred, y);

        let jll = fitted_clf.joint_log_likelihood(&x);
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
        let y = array![1., 1., 1., 2., 2., 2.];

        let mut clf = GaussianNb::new();
        let fitted_clf = clf.fit(&x, &y).unwrap();

        let expected = array![0.5, 0.5];

        assert_eq!(fitted_clf.priors, expected);
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
        let y = array![1., 1., 1., 2., 2., 2.];

        let priors = array![0.3, 0.7];

        let mut clf = GaussianNb::new().priors(priors.clone());
        let fitted_clf = clf.fit(&x, &y).unwrap();

        assert_eq!(fitted_clf.priors, priors);
    }
}
