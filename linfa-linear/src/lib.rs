//! # Linear Models
//!
//! `linfa-linear` aims to provide pure Rust implementations of
//! popular linear regression algorithms.
//!
//! ## The Big Picture
//!
//! `linfa-linear` is a crate in the [`linfa`](https://crates.io/crates/linfa)
//! ecosystem, a wider effort to bootstrap a toolkit for classical
//! Machine Learning implemented in pure Rust, kin in spirit to
//! Python's `scikit-learn`.
//!
//! ## Current state
//!
//! Right now `linfa-linear` provides ordinary least squares regression.
//!
//! ## Examples
//!
//! There is an usage example in the `examples/diabetes.rs` file, to run it
//! run
//!
//! ```bash
//! $ cargo run --features openblas --examples diabetes
//! ```

#![allow(non_snake_case)]
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar, Solve};
use ndarray_stats::SummaryStatisticsExt;
use num_traits::float::Float;

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
pub struct LinearRegression {
    options: Options,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Options {
    None,
    WithIntercept,
    WithInterceptAndNormalize,
}

impl Options {
    fn should_use_intercept(&self) -> bool {
        *self == Options::WithIntercept || *self == Options::WithInterceptAndNormalize
    }

    fn should_normalize(&self) -> bool {
        *self == Options::WithInterceptAndNormalize
    }
}

/// A fitted linear regression model which can be used for making predictions.
pub struct FittedLinearRegression<A> {
    intercept: A,
    params: Array1<A>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        LinearRegression::new()
    }
}

/// Configure and fit a linear regression model
impl LinearRegression {
    /// Create a default linear regression model.
    ///
    /// By default, an intercept will be fitted. To disable fitting an
    /// intercept, call `.with_intercept(false)` before calling `.fit()`.
    ///
    /// To additionally normalize the feature matrix before fitting, call
    /// `fit_intercept_and_normalize()` before calling `fit()`. The feature
    /// matrix will not be normalized by default.
    pub fn new() -> LinearRegression {
        LinearRegression {
            options: Options::WithIntercept,
        }
    }

    /// Configure the linear regression model to fit an intercept.
    /// Defaults to `true` if not set.
    pub fn with_intercept(mut self, with_intercept: bool) -> Self {
        if with_intercept {
            self.options = Options::WithIntercept;
        } else {
            self.options = Options::None;
        }
        self
    }

    /// Configure the linear regression model to fit an intercept and to
    /// normalize the feature matrix before fitting it.
    ///
    /// Normalizing the feature matrix is generally recommended to improve
    /// numeric stability unless features have already been normalized or
    /// are all within in a small range and all features are of similar size.
    ///
    /// Normalization implies fitting an intercept.
    pub fn with_intercept_and_normalize(mut self) -> Self {
        self.options = Options::WithInterceptAndNormalize;
        self
    }

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
    pub fn fit<A, B, C>(
        &self,
        X: &ArrayBase<B, Ix2>,
        y: &ArrayBase<C, Ix1>,
    ) -> Result<FittedLinearRegression<A>, String>
    where
        A: Lapack + Scalar + ScalarOperand + Float,
        B: Data<Elem = A>,
        C: Data<Elem = A>,
    {
        let (n_samples, _) = X.dim();

        // Check that our inputs have compatible shapes
        assert_eq!(y.dim(), n_samples);

        if self.options.should_use_intercept() {
            // If we are fitting the intercept, we first center X and y,
            // compute the models parameters based on the centered X and y
            // and the intercept as the residual of fitted parameters applied
            // to the X_offset and y_offset
            let X_offset: Array1<A> = X
                .mean_axis(Axis(0))
                .ok_or_else(|| String::from("cannot compute mean of X"))?;
            let X_centered: Array2<A> = X - &X_offset;
            let y_offset: A = y.mean().ok_or_else(|| String::from("cannot compute mean of y"))?;
            let y_centered: Array1<A> = y - y_offset;
            let params: Array1<A> =
                compute_params(&X_centered, &y_centered, self.options.should_normalize())?;
            let intercept: A = y_offset - X_offset.dot(&params);
            Ok(FittedLinearRegression {
                intercept,
                params,
            })
        } else {
            Ok(FittedLinearRegression {
                intercept: A::from(0).unwrap(),
                params: solve_normal_equation(X, y)?,
            })
        }
    }
}

/// Compute the parameters for the linear regression model with
/// or without normalization.
fn compute_params<A, B, C>(
    X: &ArrayBase<B, Ix2>,
    y: &ArrayBase<C, Ix1>,
    normalize: bool,
) -> Result<Array1<A>, String>
where
    A: Scalar + Lapack + Float,
    B: Data<Elem = A>,
    C: Data<Elem = A>,
{
    if normalize {
        let scale: Array1<A> = X.map_axis(Axis(0), |column| column.central_moment(2).unwrap());
        let X: Array2<A> = X / &scale;
        let mut params: Array1<A> = solve_normal_equation(&X, y)?;
        params /= &scale;
        Ok(params)
    } else {
        solve_normal_equation(X, y)
    }
}

/// Solve the overconstrained model Xb = y by solving X^T X b = X^t y,
/// this is (mathematically, not numerically) equivalent to computing
/// the solution with the Moore-Penrose pseudo-inverse.
fn solve_normal_equation<A, B, C>(
    X: &ArrayBase<B, Ix2>,
    y: &ArrayBase<C, Ix1>,
) -> Result<Array1<A>, String>
where
    A: Lapack + Scalar,
    B: Data<Elem = A>,
    C: Data<Elem = A>,
{
    let rhs = X.t().dot(y);
    let linear_operator = X.t().dot(X);
    linear_operator
        .solve_into(rhs)
        .map_err(|err| format! {"{}", err})

}

/// View the fitted parameters and make predictions with a fitted
/// linear regresssion model.
impl<A: Scalar + ScalarOperand> FittedLinearRegression<A> {
    /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
    /// `predict` returns the target variable according to linear model
    /// learned from the training data distribution.
    pub fn predict(&self, X: &Array2<A>) -> Array1<A> {
        X.dot(&self.params) + self.intercept
    }

    /// Get the fitted parameters
    pub fn params(&self) -> &Array1<A> {
        &self.params
    }

    /// Get the fitted intercept, 0. if no intercept was fitted
    pub fn intercept(&self) -> A {
        self.intercept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::abs_diff_eq;
    use ndarray::{array, s, Array1, Array2};

    #[test]
    fn fits_a_line_through_two_dots() {
        let lin_reg = LinearRegression::new();
        let A: Array2<f64> = array![[0.], [1.]];
        let b: Array1<f64> = array![1., 2.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let result = model.predict(&A);

        abs_diff_eq!(result, &array![1., 2.], epsilon = 1e-12);
    }

    /// When `with_intercept` is set to false, the
    /// fitted line runs through the origin. For a perfect
    /// fit we only need to provide one point.
    #[test]
    fn without_intercept_fits_line_through_origin() {
        let lin_reg = LinearRegression::new().with_intercept(false);
        let A: Array2<f64> = array![[1.]];
        let b: Array1<f64> = array![1.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let result = model.predict(&array![[0.], [1.]]);

        abs_diff_eq!(result, &array![0., 1.], epsilon = 1e-12);
    }

    /// We can't fit a line through two points without fitting the
    /// intercept in general. In this case we should find the solution
    /// that minimizes the squares. Fitting a line with intercept through
    /// the points (-1, 1), (1, 1) has the least-squares solution
    /// f(x) = 0
    #[test]
    fn fits_least_squares_line_through_two_dots() {
        let lin_reg = LinearRegression::new().with_intercept(false);
        let A: Array2<f64> = array![[-1.], [1.]];
        let b: Array1<f64> = array![1., 1.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let result = model.predict(&A);

        abs_diff_eq!(result, &array![0., 0.], epsilon = 1e-12);
    }

    /// We can't fit a line through three points in general
    /// - in this case we should find the solution that minimizes
    /// the squares. Fitting a line with intercept through the
    /// points (0, 0), (1, 0), (2, 2) has the least-squares solution
    /// f(x) = -1./3. + x
    #[test]
    fn fits_least_squares_line_through_three_dots() {
        let lin_reg = LinearRegression::new();
        let A: Array2<f64> = array![[0.], [1.], [2.]];
        let b: Array1<f64> = array![0., 0., 2.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let actual = model.predict(&A);

        abs_diff_eq!(actual, array![-1. / 3., 2. / 3., 5. / 3.], epsilon = 1e-12);
    }

    /// Check that the linear regression prefectly fits three datapoints for
    /// the model
    /// f(x) = (x + 1)^2 = x^2 + 2x + 1
    #[test]
    fn fits_three_parameters_through_three_dots() {
        let lin_reg = LinearRegression::new();
        let A: Array2<f64> = array![[0., 0.], [1., 1.], [2., 4.]];
        let b: Array1<f64> = array![1., 4., 9.];
        let model = lin_reg.fit(&A, &b).unwrap();

        abs_diff_eq!(model.params(), &array![2., 1.], epsilon = 1e-12);
        abs_diff_eq!(model.intercept(), &1., epsilon = 1e-12);
    }

    /// Check that the linear regression prefectly fits four datapoints for
    /// the model
    /// f(x) = (x + 1)^3 = x^3 + 3x^2 + 3x + 1
    #[test]
    fn fits_four_parameters_through_four_dots() {
        let lin_reg = LinearRegression::new();
        let A: Array2<f64> = array![[0., 0., 0.], [1., 1., 1.], [2., 4., 8.], [3., 9., 27.]];
        let b: Array1<f64> = array![1., 8., 27., 64.];
        let model = lin_reg.fit(&A, &b).unwrap();

        abs_diff_eq!(model.params(), &array![3., 3., 1.], epsilon = 1e-12);
        abs_diff_eq!(model.intercept(), &1., epsilon = 1e-12);
    }

    /// Check that the linear regression prefectly fits three datapoints for
    /// the model
    /// f(x) = (x + 1)^2 = x^2 + 2x + 1
    #[test]
    fn fits_three_parameters_through_three_dots_f32() {
        let lin_reg = LinearRegression::new();
        let A: Array2<f32> = array![[0., 0.], [1., 1.], [2., 4.]];
        let b: Array1<f32> = array![1., 4., 9.];
        let model = lin_reg.fit(&A, &b).unwrap();

        abs_diff_eq!(model.params(), &array![2., 1.], epsilon = 1e-4);
        abs_diff_eq!(model.intercept(), &1., epsilon = 1e-6);
    }

    /// Check that the linear regression prefectly fits four datapoints for
    /// the model
    /// f(x) = (x + 1)^3 = x^3 + 3x^2 + 3x + 1
    /// when normalization is enabled
    #[test]
    fn fits_four_parameters_through_four_dots_with_normalization() {
        let lin_reg = LinearRegression::new().with_intercept_and_normalize();
        let A: Array2<f64> = array![[0., 0., 0.], [1., 1., 1.], [2., 4., 8.], [3., 9., 27.]];
        let b: Array1<f64> = array![1., 8., 27., 64.];
        let model = lin_reg.fit(&A, &b).unwrap();

        abs_diff_eq!(model.params(), &array![3., 3., 1.], epsilon = 1e-12);
        abs_diff_eq!(model.intercept(), 1., epsilon = 1e-12);
    }

    /// Check that the linear regression model works with both owned and view
    /// representations of arrays
    #[test]
    fn works_with_viewed_and_owned_representations() {
        let lin_reg = LinearRegression::new().with_intercept_and_normalize();
        let A: Array2<f64> = array![[0., 0., 0.], [1., 1., 1.], [2., 4., 8.], [3., 9., 27.]];
        let b: Array1<f64> = array![1., 8., 27., 64.];
        let A_view = A.slice(s![.., ..]);
        let b_view = b.slice(s![..]);

        let model1 = lin_reg.fit(&A, &b).expect("can't fit owned arrays");
        let model2 = lin_reg
            .fit(&A_view, &b)
            .expect("can't fit feature view with owned target");
        let model3 = lin_reg
            .fit(&A, &b_view)
            .expect("can't fit owned features with target view");
        let model4 = lin_reg
            .fit(&A_view, &b_view)
            .expect("can't fit viewed arrays");

        assert_eq!(model1.params(), model2.params());
        assert_eq!(model2.params(), model3.params());
        assert_eq!(model3.params(), model4.params());

        abs_diff_eq!(model1.intercept(), model2.intercept());
        abs_diff_eq!(model2.intercept(), model3.intercept());
        abs_diff_eq!(model3.intercept(), model4.intercept());
    }
}
