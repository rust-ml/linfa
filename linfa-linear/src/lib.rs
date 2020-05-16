#![allow(non_snake_case)]
use num_traits::float::Float;
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar, Solve};
use ndarray_stats::SummaryStatisticsExt;

/// The simple linear regression model is
///     y = bX + e  where e ~ N(0, sigma^2 * I)
/// In probabilistic terms this corresponds to
///     y - bX ~ N(0, sigma^2 * I)
///     y | X, b ~ N(bX, sigma^2 * I)
/// The loss for the model is simply the squared error between the model
/// predictions and the true values:
///     Loss = ||y - bX||^2
/// The maximum likelihood estimation for the model parameters `beta` can be computed
/// in closed form via the normal equation:
///     b = (X^T X)^{-1} X^T y
/// where (X^T X)^{-1} X^T is known as the pseudoinverse or Moore-Penrose inverse.
///
/// Adapted from: https://github.com/ddbourgin/numpy-ml
pub struct LinearRegression {
    options: Options,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Options {
    None,
    FitIntercept,
    FitInterceptAndNormalize
}

pub struct FittedLinearRegression<A> {
    intercept: A,
    params: Array1<A>,
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression {
            options: Options::None
        }
    }

    pub fn fit_intercept(mut self) -> Self {
        self.options = Options::FitIntercept;
        self
    }

    pub fn fit_intercept_and_normalize(mut self) -> Self {
        self.options = Options::FitInterceptAndNormalize;
        self
    }

    /// Given:
    /// - an input matrix `X`, with shape `(n_samples, n_features)`;
    /// - a target variable `y`, with shape `(n_samples,)`;
    /// `fit` tunes the `beta` parameter of the linear regression model
    /// to match the training data distribution.
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

            // FIXME: double check this!
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

fn fit_intercept(options: Options) -> bool {
    options == Options::FitIntercept || options == Options::FitInterceptAndNormalize
}

fn normalize(options: Options) -> bool {
    options == Options::FitInterceptAndNormalize
}

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

impl<A: Scalar + ScalarOperand> FittedLinearRegression<A> {
    /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
    /// `predict` returns the target variable according to linear model
    /// learned from the training data distribution.
    pub fn predict(&self, X: &Array2<A>) -> Array1<A> {
        X.dot(&self.params) + self.intercept
    }

    pub fn get_params(&self) -> &Array1<A> {
        &self.params
    }

    pub fn get_intercept(&self) -> A {
        self.intercept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use std::cmp::PartialOrd;

    fn check_approx_eq<A: Scalar + PartialOrd>(lhs: &Array1<A>, rhs: &Array1<A>) {
        let diff = lhs - rhs;
        let sq_diff = diff.dot(&diff);
        assert!(sq_diff < A::from(1e-8).unwrap());
    }

    #[test]
    fn fits_a_line_through_two_dots() {
        let lin_reg = LinearRegression::new().fit_intercept();
        let A: Array2<f64> = array![[0.], [1.]];
        let b: Array1<f64> = array![1., 2.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let result = model.predict(&A);
        assert_eq!(result, array![1., 2.]);
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
        assert_eq!(result, array![0., 1.]);
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
        assert_eq!(result, array![0., 0.]);
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
        let expected = array![-1. / 3., 2. / 3., 5. / 3.];
        check_approx_eq(&actual, &expected);
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
        let expected_params: Array1<f64> = array![2., 1.];
        check_approx_eq(model.get_params(), &expected_params);
        assert!((model.get_intercept() - 1.) * (model.get_intercept() - 1.) < 1e-12);
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
        let expected_params: Array1<f64> = array![3., 3., 1.];
        check_approx_eq(model.get_params(), &expected_params);
        assert!((model.get_intercept() - 1.) * (model.get_intercept() - 1.) < 1e-12);
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
        let expected_params: Array1<f32> = array![2., 1.];
        check_approx_eq(model.get_params(), &expected_params);
        assert!((model.get_intercept() - 1.) * (model.get_intercept() - 1.) < 1e-12);
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
        let expected_params: Array1<f64> = array![3., 3., 1.];
        check_approx_eq(model.get_params(), &expected_params);
        assert!((model.get_intercept() - 1.) * (model.get_intercept() - 1.) < 1e-12);
    }
}
