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
use num_traits::float::Float;
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar, Solve};
use ndarray_stats::SummaryStatisticsExt;



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
    FitIntercept,
    FitInterceptAndNormalize
}

fn fit_intercept(options: Options) -> bool {
    options == Options::FitIntercept || options == Options::FitInterceptAndNormalize
}

fn normalize(options: Options) -> bool {
    options == Options::FitInterceptAndNormalize
}


/// A fitted linear regression model which can be used for making predictions.
pub struct FittedLinearRegression<A> {
    intercept: A,
    params: Array1<A>,
}


/// Configure and fit a linear regression model
impl LinearRegression {
    /// Create a default linear regression model. By default, no intercept
    /// will be fitted and the feature matrix will not be normalized.
    /// To change this, call `fit_intercept()` or 
    /// `fit_intercept_and_normalize()` before calling `fit()`.
    pub fn new() -> LinearRegression {
        LinearRegression {
            options: Options::None
        }
    }

    /// Configure the linear regression model to fit an intercept.
    pub fn fit_intercept(mut self) -> Self {
        self.options = Options::FitIntercept;
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
    pub fn fit_intercept_and_normalize(mut self) -> Self {
        self.options = Options::FitInterceptAndNormalize;
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
    pub fn fit<A>(&self, X: &Array2<A>, y: &Array1<A>) -> Result<FittedLinearRegression<A>, String>
    where
        A: Lapack + Scalar + ScalarOperand + Float,
    {
        let (n_samples, _) = X.dim();

        // Check that our inputs have compatible shapes
        assert_eq!(y.dim(), n_samples);

        if fit_intercept(self.options) {
            // If we are fitting the intercept, we first center X and y,
            // compute the models parameters based on the centered X and y
            // and the intercept as the residual of fitted parameters applied
            // to the X_offset and y_offset
            let X_offset: Array1<A> = X
                .mean_axis(Axis(0))
                .ok_or(String::from("cannot compute mean of X"))?;
            let X_centered: Array2<A> = X - &X_offset;
            let y_offset: A = y.mean().ok_or(String::from("cannot compute mean of y"))?;
            let y_centered: Array1<A> = y - y_offset;
            let params: Array1<A> = compute_params(&X_centered, &y_centered, normalize(self.options))?;
            let intercept: A = y_offset - X_offset.dot(&params);
            return Ok(FittedLinearRegression {
                intercept: intercept,
                params: params,
            });
        } else {
            return Ok(FittedLinearRegression {
                intercept: A::from(0).unwrap(),
                params: solve_normal_equation(X, y)?,
            });
        };
    }
}

/// Compute the parameters for the linear regression model with
/// or without normalization.
fn compute_params<A>(X: &Array2<A>, y: &Array1<A>, normalize: bool) 
    -> Result<Array1<A>, String> 
    where
        A: Lapack + Scalar + Float,
        Array2<A>: Solve<A>
{
    if normalize {
        let scale: Array1<A> = X.map_axis(Axis(0), |column| column.central_moment(2).unwrap());
        let X = X / &scale;
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
fn solve_normal_equation<A>(X: &Array2<A>, y: &Array1<A>) -> Result<Array1<A>, String>
where
    A: Lapack + Scalar,
    Array2<A>: Solve<A>,
{
    let rhs = X.t().dot(y);
    let linear_operator = X.t().dot(X);
    linear_operator
        .solve_into(rhs)
        .or_else(|err| Err(format! {"{}", err}))
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
    pub fn get_params(&self) -> &Array1<A> {
        &self.params
    }

    /// Get the fitted intercept, 0. if no intercept was fitted
    pub fn get_intercept(&self) -> A {
        self.intercept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::AbsDiffEq;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn fits_a_line_through_two_dots() {
        let lin_reg = LinearRegression::new().fit_intercept();
        let A: Array2<f64> = array![[0.], [1.]];
        let b: Array1<f64> = array![1., 2.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let result = model.predict(&A);

        assert!(result.abs_diff_eq(&array![1., 2.], 1e-12));
    }

    /// When `fit_intercept` is not set (the default), the
    /// fitted line runs through the origin. For a perfect
    /// fit we only need to provide one point.
    #[test]
    fn without_intercept_fits_line_through_origin() {
        let lin_reg = LinearRegression::new();
        let A: Array2<f64> = array![[1.]];
        let b: Array1<f64> = array![1.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let result = model.predict(&array![[0.], [1.]]);

        assert!(result.abs_diff_eq(&array![0., 1.], 1e-12));
    }

    /// We can't fit a line through two points without fitting the
    /// intercept in general. In this case we should find the solution
    /// that minimizes the squares. Fitting a line with intercept through
    /// the points (-1, 1), (1, 1) has the least-squares solution
    /// f(x) = 0
    #[test]
    fn fits_least_squares_line_through_two_dots() {
        let lin_reg = LinearRegression::new();
        let A: Array2<f64> = array![[-1.], [1.]];
        let b: Array1<f64> = array![1., 1.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let result = model.predict(&A);

        assert!(result.abs_diff_eq(&array![0., 0.], 1e-12));
    }

    /// We can't fit a line through three points in general
    /// - in this case we should find the solution that minimizes
    /// the squares. Fitting a line with intercept through the
    /// points (0, 0), (1, 0), (2, 2) has the least-squares solution
    /// f(x) = -1./3. + x
    #[test]
    fn fits_least_squares_line_through_three_dots() {
        let lin_reg = LinearRegression::new().fit_intercept();
        let A: Array2<f64> = array![[0.], [1.], [2.]];
        let b: Array1<f64> = array![0., 0., 2.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let actual = model.predict(&A);

        assert!(actual.abs_diff_eq(&array![-1. / 3., 2. / 3., 5. / 3.], 1e-12));
    }

    /// Check that the linear regression prefectly fits three datapoints for
    /// the model
    /// f(x) = (x + 1)^2 = x^2 + 2x + 1
    #[test]
    fn fits_three_parameters_through_three_dots() {
        let lin_reg = LinearRegression::new().fit_intercept();
        let A: Array2<f64> = array![[0., 0.], [1., 1.], [2., 4.]];
        let b: Array1<f64> = array![1., 4., 9.];
        let model = lin_reg.fit(&A, &b).unwrap();

        assert!(model.get_params().abs_diff_eq(&array![2., 1.], 1e-12));
        assert!(model.get_intercept().abs_diff_eq(&1., 1e-12));
    }

    /// Check that the linear regression prefectly fits four datapoints for
    /// the model
    /// f(x) = (x + 1)^3 = x^3 + 3x^2 + 3x + 1
    #[test]
    fn fits_four_parameters_through_four_dots() {
        let lin_reg = LinearRegression::new().fit_intercept();
        let A: Array2<f64> = array![[0., 0., 0.], [1., 1., 1.], [2., 4., 8.], [3., 9., 27.]];
        let b: Array1<f64> = array![1., 8., 27., 64.];
        let model = lin_reg.fit(&A, &b).unwrap();
        
        assert!(model.get_params().abs_diff_eq(&array![3., 3., 1.], 1e-12));
        assert!(model.get_intercept().abs_diff_eq(&1., 1e-12));
    }

    /// Check that the linear regression prefectly fits three datapoints for
    /// the model
    /// f(x) = (x + 1)^2 = x^2 + 2x + 1
    #[test]
    fn fits_three_parameters_through_three_dots_f32() {
        let lin_reg = LinearRegression::new().fit_intercept();
        let A: Array2<f32> = array![[0., 0.], [1., 1.], [2., 4.]];
        let b: Array1<f32> = array![1., 4., 9.];
        let model = lin_reg.fit(&A, &b).unwrap();

        assert!(model.get_params().abs_diff_eq(&array![2., 1.], 1e-4));
        assert!(model.get_intercept().abs_diff_eq(&1., 1e-6));
    }

    /// Check that the linear regression prefectly fits four datapoints for
    /// the model
    /// f(x) = (x + 1)^3 = x^3 + 3x^2 + 3x + 1
    /// when normalization is enabled
    #[test]
    fn fits_four_parameters_through_four_dots_with_normalization() {
        let lin_reg = LinearRegression::new().fit_intercept_and_normalize();
        let A: Array2<f64> = array![[0., 0., 0.], [1., 1., 1.], [2., 4., 8.], [3., 9., 27.]];
        let b: Array1<f64> = array![1., 8., 27., 64.];
        let model = lin_reg.fit(&A, &b).unwrap();

        assert!(model.get_params().abs_diff_eq(&array![3., 3., 1.], 1e-12));
        assert!(model.get_intercept().abs_diff_eq(&1., 1e-12));
    }
}
