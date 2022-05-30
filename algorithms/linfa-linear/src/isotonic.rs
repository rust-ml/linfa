//! Isotonic
#![allow(non_snake_case)]
use crate::error::{LinearError, Result};
use ndarray::{stack, Array1, ArrayBase, Axis, Data, Ix1, Ix2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use linfa::dataset::{AsSingleTargets, DatasetBase};
use linfa::traits::{Fit, PredictInplace};

pub trait Float: linfa::Float {}
impl Float for f32 {}
impl Float for f64 {}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
/// An isotonic regression model.
///
/// IsotonicRegression solves an isotonic regression problem using the pool
/// adjacent violators algorithm.
///
/// /// ## Examples
///
/// Here's an example on how to train an isotonic regression model on
/// the first feature from the `diabetes` dataset.
/// ```rust
/// use linfa::{traits::Fit, traits::Predict, Dataset};
/// use linfa_linear::IsotonicRegression;
/// use linfa::prelude::SingleTargetRegression;
///
/// let diabetes = linfa_datasets::diabetes();
/// let dataset = diabetes.feature_iter().next().unwrap(); // get first 1D feature
/// let model = IsotonicRegression::default().fit(&dataset).unwrap();
/// let pred = model.predict(&dataset);
/// let r2 = pred.r2(&dataset).unwrap();
/// println!("r2 from prediction: {}", r2);
/// ```
pub struct IsotonicRegression {}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
/// A fitted isotonic regression model which can be used for making predictions.
pub struct FittedIsotonicRegression<F> {
    regressor: Array1<F>,
    response: Array1<F>,
}

/// Configure and fit a isotonic regression model
impl IsotonicRegression {
    /// Create a default isotonic regression model.
    pub fn new() -> IsotonicRegression {
        IsotonicRegression {}
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = F>>
    Fit<ArrayBase<D, Ix2>, T, LinearError<F>> for IsotonicRegression
{
    type Object = FittedIsotonicRegression<F>;

    /// Fit an isotonic regression model given a feature matrix `X` and a target
    /// variable `y`.
    ///
    /// The feature matrix `X` must have shape `(n_samples, 1)`
    ///
    /// The target variable `y` must have shape `(n_samples)`
    ///
    /// Returns a `FittedIsotonicRegression` object which contains the fitted
    /// parameters and can be used to `predict` values of the target variable
    /// for new feature values.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object, F> {
        let X = dataset.records();
        let (n_samples, dim) = X.dim();
        let y = dataset.as_single_targets();

        // Check the input dimension
        assert_eq!(dim, 1, "The input dimension must be 1.");

        // Check that our inputs have compatible shapes
        assert_eq!(y.dim(), n_samples);

        // use correlation for determining relationship between x & y
        let x = X.column(0);
        let rho = DatasetBase::from(stack![Axis(1), x, y]).pearson_correlation();
        let increasing = rho.get_coeffs()[0] >= F::zero();

        // sort data
        let indices = argsort_by(&x, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));

        // PVA algorithm
        let mut J: Vec<(F, F, Vec<usize>)> = indices
            .iter()
            .map(|&i| (y[i], F::cast(dataset.weight_for(i)), vec![i]))
            .collect();
        if !increasing {
            J.reverse();
        };

        let mut i: usize = 0;
        while i < (J.len() - 1) {
            let B_zero = &J[i];
            let B_plus = &J[i + 1];
            if B_zero.0 <= B_plus.0 {
                i += 1;
            } else {
                let w0 = B_zero.1 + B_plus.1;
                let v0 = (B_zero.0 * B_zero.1 + B_plus.0 * B_plus.1) / w0;
                let mut i0 = B_zero.2.to_vec();
                i0.extend(&(B_plus.2));
                J[i] = (v0, w0, i0);
                J.remove(i + 1);
                let idx = i;
                while i > 0 {
                    let B_minus = &J[i - 1];
                    if v0 <= B_minus.0 {
                        let ww = w0 + B_minus.1;
                        let vv = (v0 * w0 + B_minus.0 * B_minus.1) / ww;
                        let mut ii = J[idx].2.to_vec();
                        ii.extend(&(B_minus.2));
                        J[i] = (vv, ww, ii);
                        J.remove(i - 1);
                        i -= 1;
                    } else {
                        break;
                    }
                }
            }
        }
        if !increasing {
            J.reverse();
        };

        // Form model parameters
        let params: (Vec<F>, Vec<F>) = J
            .iter()
            .map(|e| {
                (
                    e.0,
                    x.select(Axis(0), &e.2)
                        .into_iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater))
                        .unwrap(),
                )
            })
            .unzip();
        let regressor = Array1::from_vec(params.1);
        let response = Array1::from_vec(params.0);
        Ok(FittedIsotonicRegression {
            regressor,
            response,
        })
    }
}

fn argsort_by<S, F>(arr: &ArrayBase<S, Ix1>, mut compare: F) -> Vec<usize>
where
    S: Data,
    F: FnMut(&S::Elem, &S::Elem) -> Ordering,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_unstable_by(move |&i, &j| compare(&arr[i], &arr[j]));
    indices
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<F>>
    for FittedIsotonicRegression<F>
{
    /// Given an input matrix `X`, with shape `(n_samples, 1)`,
    /// `predict` returns the target variable according to linear model
    /// learned from the training data distribution.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<F>) {
        let (n_samples, dim) = x.dim();

        // Check the input dimension
        assert_eq!(dim, 1, "The input dimension must be 1.");

        // Check that our inputs have compatible shapes
        assert_eq!(
            n_samples,
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let regressor = &self.regressor;
        let n = regressor.len();
        let x_min = regressor[0];
        let x_max = regressor[n - 1];

        let response = &self.response;
        let y_min = response[0];
        let y_max = response[n - 1];

        // calculate a piecewise linear approximation of response
        for (i, row) in x.rows().into_iter().enumerate() {
            let val = row[0];
            if val >= x_max {
                y[i] = y_max;
            } else if val <= x_min {
                y[i] = y_min;
            } else {
                let res = regressor.into_iter().position(|x| x >= &val);
                if res.is_some() {
                    let j = res.unwrap();
                    if val <= regressor[j] && j < n {
                        let x_scale = (val - regressor[j - 1]) / (regressor[j] - regressor[j - 1]);
                        y[i] = response[j - 1] + x_scale * (response[j] - response[j - 1]);
                    } else {
                        y[i] = y_min;
                    }
                }
            }
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        Array1::zeros(x.nrows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::{traits::Predict, Dataset};
    use ndarray::array;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<FittedIsotonicRegression<f64>>();
        has_autotraits::<IsotonicRegression>();
        has_autotraits::<LinearError<f64>>();
    }

    #[test]
    fn dimension_mismatch() {
        let reg = IsotonicRegression::new();
        let dataset = Dataset::new(array![[3.3f64, 0.], [3.3, 0.]], array![4., 5.]);
        let res = std::panic::catch_unwind(|| reg.fit(&dataset));
        assert!(res.is_err());
    }

    #[test]
    fn length_mismatch() {
        let reg = IsotonicRegression::new();
        let dataset = Dataset::new(array![[3.3f64, 0.], [3.3, 0.]], array![4., 5., 6.]);
        let res = std::panic::catch_unwind(|| reg.fit(&dataset));
        assert!(res.is_err());
    }

    #[test]
    fn best_example1() {
        let reg = IsotonicRegression::new();
        let dataset = Dataset::new(
            array![[3.3f64], [3.3], [3.3], [6.], [7.5], [7.5]],
            array![4., 5., 1., 6., 8., 7.0],
        );
        let model = reg.fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.regressor, &array![3.3, 6., 7.5], epsilon = 1e-12);
        assert_abs_diff_eq!(
            model.response,
            &array![10.0 / 3.0, 6., 7.5],
            epsilon = 1e-12
        );

        let result = model.predict(dataset.records());
        assert_abs_diff_eq!(
            result,
            &array![10. / 3., 10. / 3., 10. / 3., 6., 7.5, 7.5],
            epsilon = 1e-12
        );

        let xs = array![[2.0f64], [5.], [7.0], [9.0]];
        let result = model.predict(&xs);
        assert_abs_diff_eq!(
            result,
            &array![10. / 3., 5.01234567901234, 7., 7.5],
            epsilon = 1e-12
        );
    }

    #[test]
    fn decr_best_example1() {
        let reg = IsotonicRegression::new();
        let dataset = Dataset::new(
            array![[7.5f64], [7.5], [6.], [3.3], [3.3], [3.3]],
            array![4., 5., 1., 6., 8., 7.0],
        );
        let model = reg.fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.regressor, &array![3.3, 7.5], epsilon = 1e-12);
        assert_abs_diff_eq!(model.response, &array![7.0, 10.0 / 3.0], epsilon = 1e-12);

        let result = model.predict(dataset.records());
        assert_abs_diff_eq!(
            result,
            &array![10. / 3., 10. / 3., 4.64285714285714, 7., 7., 7.],
            epsilon = 1e-12
        );
    }
}
