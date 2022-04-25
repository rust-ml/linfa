//! Ordinary Least Squares
#![allow(non_snake_case)]
use crate::error::{LinearError, Result};
#[cfg(feature = "blas")]
use linfa::dataset::{WithLapack, WithoutLapack};
use linfa::Float;
use ndarray::{concatenate, s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
#[cfg(feature = "blas")]
use ndarray_linalg::LeastSquaresSvdInto;
#[cfg(not(feature = "blas"))]
use ndarray_linalg_rs::qr::LeastSquaresQrInto;
use serde::{Deserialize, Serialize};

use linfa::dataset::{AsSingleTargets, DatasetBase};
use linfa::traits::{Fit, PredictInplace};

#[derive(Serialize, Deserialize)]
/// An ordinary least squares linear regression model.
///
/// LinearRegression fits a linear model to minimize the residual sum of
/// squares between the observed targets in the dataset, and the targets
/// predicted by the linear approximation.
///
/// Ordinary least squares regression solves the overconstrainted model
///
/// y = Ax + b
///
/// by finding x and b which minimize the L_2 norm ||y - Ax - b||_2.
///
/// It currently uses the [Moore-Penrose pseudo-inverse]()
/// to solve y - b = Ax.
///
/// /// ## Examples
///
/// Here's an example on how to train a linear regression model on the `diabetes` dataset
/// ```rust
/// use linfa::traits::{Fit, Predict};
/// use linfa_linear::LinearRegression;
/// use linfa::prelude::SingleTargetRegression;
///
/// let dataset = linfa_datasets::diabetes();
/// let model = LinearRegression::default().fit(&dataset).unwrap();
/// let pred = model.predict(&dataset);
/// let r2 = pred.r2(&dataset).unwrap();
/// println!("r2 from prediction: {}", r2);
/// ```
pub struct LinearRegression {
    fit_intercept: bool,
}

#[derive(Serialize, Deserialize)]
/// A fitted linear regression model which can be used for making predictions.
pub struct FittedLinearRegression<F> {
    intercept: F,
    params: Array1<F>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        LinearRegression::new()
    }
}

/// Configure and fit a linear regression model
impl LinearRegression {
    /// Create a default linear regression model.
    /// By default, an intercept will be fitted.
    pub fn new() -> LinearRegression {
        LinearRegression {
            fit_intercept: true,
        }
    }

    /// Configure the linear regression model to fit an intercept.
    pub fn with_intercept(mut self, intercept: bool) -> Self {
        self.fit_intercept = intercept;
        self
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = F>>
    Fit<ArrayBase<D, Ix2>, T, LinearError<F>> for LinearRegression
{
    type Object = FittedLinearRegression<F>;

    /// Fit a linear regression model given a feature matrix `X` and a target
    /// variable `y`.
    ///
    /// The feature matrix `X` must have shape `(n_samples, n_features)`
    ///
    /// The target variable `y` must have shape `(n_samples)`
    ///
    /// Returns a `FittedLinearRegression` object which contains the fitted
    /// parameters and can be used to `predict` values of the target variable
    /// for new feature values.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object, F> {
        let X = dataset.records();
        let y = dataset.as_single_targets();

        let (n_samples, _) = X.dim();

        // Check that our inputs have compatible shapes
        assert_eq!(y.dim(), n_samples);

        if self.fit_intercept {
            let X = concatenate(Axis(1), &[X.view(), Array2::ones((X.nrows(), 1)).view()]).unwrap();
            let params: Array1<F> = solve_least_squares(X, y.to_owned())?;
            let intercept = *params.last().unwrap();
            let params = params.slice(s![..params.len() - 1]).to_owned();
            Ok(FittedLinearRegression { intercept, params })
        } else {
            // `LeastSquaresSvdInto` needs a mutable reference to the data and `dataset` is taken
            // by reference. Therefore copy the problem matrix and target vector.
            let (X, y) = (X.to_owned(), y.to_owned());

            Ok(FittedLinearRegression {
                intercept: F::cast(0),
                params: solve_least_squares(X, y)?,
            })
        }
    }
}

/// Find the b that minimizes the 2-norm of X b - y
/// by using the least_squares solver from ndarray-linalg
fn solve_least_squares<F>(mut X: Array<F, Ix2>, mut y: Array<F, Ix1>) -> Result<Array1<F>, F>
where
    F: Float,
{
    // ensure that B = C
    let (X, y) = (X.view_mut(), y.view_mut());

    #[cfg(not(feature = "blas"))]
    let out = X
        .least_squares_into(y.insert_axis(Axis(1)))?
        .remove_axis(Axis(1));
    #[cfg(feature = "blas")]
    let out = X
        .with_lapack()
        .least_squares_into(y.with_lapack())
        .map(|x| x.solution)?
        .without_lapack();
    Ok(out)
}

/// View the fitted parameters and make predictions with a fitted
/// linear regresssion model.
impl<F: Float> FittedLinearRegression<F> {
    /// Get the fitted parameters
    pub fn params(&self) -> &Array1<F> {
        &self.params
    }

    /// Get the fitted intercept, 0. if no intercept was fitted
    pub fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<F>>
    for FittedLinearRegression<F>
{
    /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
    /// `predict` returns the target variable according to linear model
    /// learned from the training data distribution.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<F>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        *y = x.dot(&self.params) + self.intercept;
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
    fn fits_a_line_through_two_dots() {
        let lin_reg = LinearRegression::new();
        let dataset = Dataset::new(array![[0f64], [1.]], array![1., 2.]);
        let model = lin_reg.fit(&dataset).unwrap();
        let result = model.predict(dataset.records());

        assert_abs_diff_eq!(result, &array![1., 2.], epsilon = 1e-12);
    }

    /// When `with_intercept` is set to false, the
    /// fitted line runs through the origin. For a perfect
    /// fit we only need to provide one point.
    #[test]
    fn without_intercept_fits_line_through_origin() {
        let lin_reg = LinearRegression::new().with_intercept(false);
        let dataset = Dataset::new(array![[1.]], array![1.]);
        let model = lin_reg.fit(&dataset).unwrap();
        let result = model.predict(&array![[0.], [1.]]);

        assert_abs_diff_eq!(result, &array![0., 1.], epsilon = 1e-12);
    }

    /// We can't fit a line through two points without fitting the
    /// intercept in general. In this case we should find the solution
    /// that minimizes the squares. Fitting a line with intercept through
    /// the points (-1, 1), (1, 1) has the least-squares solution
    /// f(x) = 0
    #[test]
    fn fits_least_squares_line_through_two_dots() {
        let lin_reg = LinearRegression::new().with_intercept(false);
        let dataset = Dataset::new(array![[-1.], [1.]], array![1., 1.]);
        let model = lin_reg.fit(&dataset).unwrap();
        let result = model.predict(dataset.records());

        assert_abs_diff_eq!(result, &array![0., 0.], epsilon = 1e-12);
    }

    /// We can't fit a line through three points in general
    /// - in this case we should find the solution that minimizes
    /// the squares. Fitting a line with intercept through the
    /// points (0, 0), (1, 0), (2, 2) has the least-squares solution
    /// f(x) = -1./3. + x
    #[test]
    fn fits_least_squares_line_through_three_dots() {
        let lin_reg = LinearRegression::new();
        let dataset = Dataset::new(array![[0.], [1.], [2.]], array![0., 0., 2.]);
        let model = lin_reg.fit(&dataset).unwrap();
        let actual = model.predict(dataset.records());

        assert_abs_diff_eq!(actual, array![-1. / 3., 2. / 3., 5. / 3.], epsilon = 1e-12);
    }

    /// Check that the linear regression prefectly fits three datapoints for
    /// the model
    /// f(x) = (x + 1)^2 = x^2 + 2x + 1
    #[test]
    fn fits_three_parameters_through_three_dots() {
        let lin_reg = LinearRegression::new();
        let dataset = Dataset::new(array![[0f64, 0.], [1., 1.], [2., 4.]], array![1., 4., 9.]);
        let model = lin_reg.fit(&dataset).unwrap();

        assert_abs_diff_eq!(model.params(), &array![2., 1.], epsilon = 1e-12);
        assert_abs_diff_eq!(model.intercept(), &1., epsilon = 1e-12);
    }

    /// Check that the linear regression prefectly fits four datapoints for
    /// the model
    /// f(x) = (x + 1)^3 = x^3 + 3x^2 + 3x + 1
    #[test]
    fn fits_four_parameters_through_four_dots() {
        let lin_reg = LinearRegression::new();
        let dataset = Dataset::new(
            array![[0f64, 0., 0.], [1., 1., 1.], [2., 4., 8.], [3., 9., 27.]],
            array![1., 8., 27., 64.],
        );
        let model = lin_reg.fit(&dataset).unwrap();

        assert_abs_diff_eq!(model.params(), &array![3., 3., 1.], epsilon = 1e-12);
        assert_abs_diff_eq!(model.intercept(), &1., epsilon = 1e-12);
    }

    /// Check that the linear regression prefectly fits three datapoints for
    /// the model
    /// f(x) = (x + 1)^2 = x^2 + 2x + 1
    #[test]
    fn fits_three_parameters_through_three_dots_f32() {
        let lin_reg = LinearRegression::new();
        let dataset = Dataset::new(array![[0f64, 0.], [1., 1.], [2., 4.]], array![1., 4., 9.]);
        let model = lin_reg.fit(&dataset).unwrap();

        assert_abs_diff_eq!(model.params(), &array![2., 1.], epsilon = 1e-4);
        assert_abs_diff_eq!(model.intercept(), &1., epsilon = 1e-6);
    }

    ///// Check that the linear regression prefectly fits four datapoints for
    ///// the model
    ///// f(x) = (x + 1)^3 = x^3 + 3x^2 + 3x + 1
    ///// when normalization is enabled
    //#[test]
    //fn fits_four_parameters_through_four_dots_with_normalization() {
    //let lin_reg = LinearRegression::new().with_intercept_and_normalize();
    //let dataset = Dataset::new(
    //array![[0f64, 0., 0.], [1., 1., 1.], [2., 4., 8.], [3., 9., 27.]],
    //array![1., 8., 27., 64.],
    //);
    //let model = lin_reg.fit(&dataset).unwrap();

    //assert_abs_diff_eq!(model.params(), &array![3., 3., 1.], epsilon = 1e-12);
    //assert_abs_diff_eq!(model.intercept(), 1., epsilon = 1e-12);
    //}

    ///// Check that the linear regression model works with both owned and view
    ///// representations of arrays
    //#[test]
    //fn works_with_viewed_and_owned_representations() {
    //let lin_reg = LinearRegression::new().with_intercept_and_normalize();
    //let dataset = Dataset::new(
    //array![[0., 0., 0.], [1., 1., 1.], [2., 4., 8.], [3., 9., 27.]],
    //array![1., 8., 27., 64.],
    //);
    //let dataset_view = dataset.view();

    //let model1 = lin_reg.fit(&dataset).expect("can't fit owned arrays");
    //let model2 = lin_reg
    //.fit(&dataset_view)
    //.expect("can't fit feature view with owned target");

    //assert_eq!(model1.params(), model2.params());
    //assert_abs_diff_eq!(model1.intercept(), model2.intercept());
    //}
}
