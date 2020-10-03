use ndarray::{s, Array1, Array2, ArrayView1, Axis};
// QuantileExt trait required for `max` on ArrayBase
use ndarray_stats::QuantileExt;

use crate::error::{BayesError, Result};
use crate::Float;

struct GaussianNb<A> {
    priors: Option<Array1<A>>,
    var_smoothing: f64,
    classes: Option<Array1<A>>,
    theta: Option<Array2<A>>,
    sigma: Option<Array2<A>>,
    class_count: Option<Array1<A>>,
}

impl<A> GaussianNb<A> {
    fn new() -> Self {
        GaussianNb {
            priors: None,
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
    fn fit(&mut self, x: &Array2<A>, y: &Array1<A>) -> Result<()> {
        let unique_classes = GaussianNb::unique(&y);

        self.partial_fit(x, y, unique_classes);

        Ok(())
    }

    fn partial_fit(
        &mut self,
        x: &Array2<A>,
        y: &Array1<A>,
        classes: Array1<A>,
    ) -> Result<Array1<A>> {
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
            self.classes = Some(classes);

            let nfeatures = x.ncols();

            // unwrap is safe since assigning `self.classes` is the first step
            let nclasses = self.classes.as_ref().unwrap().len();

            self.theta = Some(Array2::zeros((nclasses, nfeatures)));
            self.sigma = Some(Array2::zeros((nclasses, nfeatures)));
            self.class_count = Some(Array1::zeros(nclasses));

            if let Some(ref priors) = self.priors {
                if priors.len() != nclasses {
                    todo!();
                }

                if (priors.sum() - A::from(1.0).unwrap()).abs() > A::from(1e-6).unwrap() {
                    todo!();
                }

                if priors.iter().any(|x| x < &A::from(0.).unwrap()) {
                    todo!();
                }
            }
        } else {
            if x.ncols() != self.theta.as_ref().unwrap().ncols() {
                todo!();
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
            todo!();
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
            *element = A::from(nclass).unwrap();
        }

        self.sigma.as_mut().unwrap().mapv_inplace(|x| x + epsilon);

        if self.priors.is_none() {
            let class_count_sum = self.class_count.as_ref().unwrap().sum();
            let temp = self
                .class_count
                .as_ref()
                .unwrap()
                .mapv(|x| x / class_count_sum);
            self.priors.as_mut().unwrap().assign(&temp);
        }

        Ok(Array1::zeros(2))
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
        let mu_new_weighted = mu_new.mapv(|x| x * A::from(count_new).unwrap());
        let mu_old_weighted = mu_old.mapv(|x| x * A::from(count_old).unwrap());
        let mu_weighted = (mu_new_weighted + mu_old_weighted).mapv(|x| x / count_total);

        // Combine old and new variance, taking into consideration the number
        // of observations. this is achieved by combining the sum of squared
        // differences
        let ssd_old = var_old.mapv(|x| x * A::from(count_old).unwrap());
        let ssd_new = var_new.mapv(|x| x * A::from(count_new).unwrap());
        let weight = A::from(count_new * count_old).unwrap() / count_total;
        let ssd_weighted = ssd_old + ssd_new + (mu_old - &mu_new).mapv(|x| weight * x.powi(2));
        let var_weighted = ssd_weighted.mapv(|x| x / count_total);

        (mu_weighted, var_weighted)
    }

    // Extract unique elements of the array in sorted order
    fn unique(y: &Array1<A>) -> Array1<A> {
        // We are identifying unique classes in y,
        // ndarray doesn't provide methods for extracting unique elements,
        // So we are converting it to a Vec
        let mut unique_classes = y.to_vec();
        unique_classes.sort_by(|x, y| x.partial_cmp(y).unwrap());
        unique_classes.dedup();

        Array1::from(unique_classes)
    }

    fn filter(x: &Array2<A>, y: &Array1<A>, ycondition: A) -> Result<Array2<A>> {
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

#[cfg(test)]
mod tests {
    use ndarray::array;
    use ndarray::s;

    use super::*;

    #[test]
    fn test_sample() {
        let mut x = array![[1., 2.], [3., 4.], [5., 6.]];
        let mut y = array![1., 2., 1.];
        let cond = 3.;
        let ans = GaussianNb::filter(&x, &y, cond).unwrap();
        //dbg!(ans.nrows());

        {
            x.row_mut(0).assign(&array![0., 0.]);
        }
        //dbg!(x);

        let element = y.get_mut(0).unwrap();
        *element = 10.;
        dbg!(y);

        assert!(false);
    }
}
