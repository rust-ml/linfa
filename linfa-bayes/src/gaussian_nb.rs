use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
// QuantileExt trait required for `max` on ArrayBase
use ndarray_stats::QuantileExt;

use crate::error::{BayesError, Result};
use crate::Float;

pub struct GaussianNb<A> {
    priors: Option<Array1<A>>,
    priors_provided: bool,
    var_smoothing: f64,
    classes: Option<Array1<A>>,
    theta: Option<Array2<A>>,
    sigma: Option<Array2<A>>,
    class_count: Option<Array1<A>>,
}

impl<A> GaussianNb<A> {
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
    fn priors(mut self, priors: Array1<A>) -> Self {
        self.priors = Some(priors);
        self.priors_provided = true;
        self
    }

    // Specifies the portion of the largest variance of all the features that
    // is added to the variance for calculation stability
    fn var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }
}

impl<A: Float + PartialEq + PartialOrd> GaussianNb<A> {
    pub fn fit(&mut self, x: &Array2<A>, y: &Array1<A>) -> Result<FittedGaussianNb<A>> {
        let unique_classes = GaussianNb::unique(&y.view());

        self.partial_fit(&x.view(), &y.view(), &unique_classes)?;
        let classifier = self.get_predictor()?;

        Ok(classifier)
    }

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

    pub fn partial_fit(
        &mut self,
        x: &ArrayView2<A>,
        y: &ArrayView1<A>,
        classes: &Array1<A>,
    ) -> Result<()> {
        // If the ratio of the variance between dimensions is too small, it will cause
        // numberical errors. We address this by artificially boosting the variance
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

            self.theta = Some(Array2::zeros((nclasses, nfeatures)));
            self.sigma = Some(Array2::zeros((nclasses, nfeatures)));
            self.class_count = Some(Array1::zeros(nclasses));

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

            let xclass = Self::filter(&x, &y, *class)?;
            let nclass = xclass.nrows();

            let (theta_new, sigma_new) = Self::update_mean_variance(
                self.class_count.as_ref().unwrap()[position],
                &self.theta.as_ref().unwrap().slice(s![position, ..]),
                &self.sigma.as_ref().unwrap().slice(s![position, ..]),
                &xclass,
            );

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

        self.sigma.as_mut().unwrap().mapv_inplace(|x| x + epsilon);

        //if self.priors.is_none() {
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

    fn update_mean_variance(
        count_old: A,
        mu_old: &ArrayView1<A>,
        var_old: &ArrayView1<A>,
        x_new: &Array2<A>,
    ) -> (Array1<A>, Array1<A>) {
        if x_new.nrows() == 0 {
            return (mu_old.to_owned(), var_old.to_owned());
        }

        let count_new = A::from(x_new.nrows()).unwrap();
        let mu_new = x_new.mean_axis(Axis(0)).unwrap();
        let var_new = x_new.var_axis(Axis(0), A::from(0.).unwrap());

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

    fn filter(x: &ArrayView2<A>, y: &ArrayView1<A>, ycondition: A) -> Result<Array2<A>> {
        let index: Vec<_> = y
            .indexed_iter()
            .filter_map(|(i, y)| {
                if (ycondition - *y).abs() < A::from(1e-6).unwrap() {
                    return Some(i);
                }
                None
            })
            .collect();

        let xsubset_vec: Vec<_> = index
            .iter()
            .map(|&i| x.slice(s![i, ..]).to_vec())
            .flatten()
            .collect();

        let xsubset = Array2::from_shape_vec((index.len(), x.ncols()), xsubset_vec)?;

        Ok(xsubset)
    }
}

#[derive(Debug)]
pub struct FittedGaussianNb<A> {
    classes: Array1<A>,
    priors: Array1<A>,
    theta: Array2<A>,
    sigma: Array2<A>,
}

impl<A: Float> FittedGaussianNb<A> {
    pub fn predict(&self, x: &Array2<A>) -> Array1<A> {
        let joint_log_liklihood = self.joint_log_likelihood(x);

        joint_log_liklihood.map_axis(Axis(1), |x| {
            let i = x.argmax().unwrap();
            *self.classes.get(i).unwrap()
        })
    }

    fn joint_log_likelihood(&self, x: &Array2<A>) -> Array2<A> {
        let mut joint_log_likelihood = Array2::zeros((x.nrows(), self.classes.len()));

        for i in 0..self.classes.len() {
            let jointi = self.priors.get(i).unwrap().ln();

            let mut nij = self
                .sigma
                .row(i)
                .mapv(|x| A::from(2.).unwrap() * A::from(std::f64::consts::PI).unwrap() * x)
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
            .exact_chunks((2, 2))
            .into_iter()
            .zip(y.exact_chunks(2).into_iter())
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
