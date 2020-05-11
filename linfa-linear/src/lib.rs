#![allow(non_snake_case)]
use ndarray::{stack, Array, Array1, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_linalg::Solve;

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
    fit_intercept: bool,
}

pub struct FittedLinearRegression {
    fit_intercept: bool,
    params: Array1<f64>,
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression {
            fit_intercept: false,
        }
    }

    pub fn with_intercept(mut self) -> Self {
        self.fit_intercept = true;
        self
    }

    /// Given:
    /// - an input matrix `X`, with shape `(n_samples, n_features)`;
    /// - a target variable `y`, with shape `(n_samples,)`;
    /// `fit` tunes the `beta` parameter of the linear regression model
    /// to match the training data distribution.
    ///
    /// `self` is modified in place, nothing is returned.
    pub fn fit<A, B>(
        &self,
        X: &ArrayBase<A, Ix2>,
        y: &ArrayBase<B, Ix1>,
    ) -> Result<FittedLinearRegression, String>
    where
        A: Data<Elem = f64>,
        B: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();

        // Check that our inputs have compatible shapes
        assert_eq!(y.dim(), n_samples);

        // If we are fitting the intercept, we need an additional column
        if self.fit_intercept {
            let dummy_column: Array<f64, _> = Array::ones((n_samples, 1));
            let X = stack(Axis(1), &[dummy_column.view(), X.view()]).unwrap();
            return Ok(FittedLinearRegression {
                fit_intercept: self.fit_intercept,
                params: solve_normal_equation(&X, y)?,
            });
        } else {
            return Ok(FittedLinearRegression {
                fit_intercept: self.fit_intercept,
                params: solve_normal_equation(X, y)?,
            });
        };
    }
}

impl FittedLinearRegression {
    /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
    /// `predict` returns the target variable according to linear model
    /// learned from the training data distribution.
    pub fn predict<A>(&self, X: &ArrayBase<A, Ix2>) -> Array1<f64>
    where
        A: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();

        // If we are fitting the intercept, we need an additional column
        if self.fit_intercept {
            let dummy_column: Array<f64, _> = Array::ones((n_samples, 1));
            let X = stack(Axis(1), &[dummy_column.view(), X.view()]).unwrap();
            self._predict(&X)
        } else {
            self._predict(X)
        }
    }

    pub fn get_params(&self) -> &Array1<f64> {
        &self.params
    }

    fn _predict<A>(&self, X: &ArrayBase<A, Ix2>) -> Array1<f64>
    where
        A: Data<Elem = f64>,
    {
        X.dot(&self.params)
    }
}

fn solve_normal_equation<A, B>(
    X: &ArrayBase<A, Ix2>,
    y: &ArrayBase<B, Ix1>,
) -> Result<Array1<f64>, String>
where
    A: Data<Elem = f64>,
    B: Data<Elem = f64>,
{
    let rhs = X.t().dot(y);
    let linear_operator = X.t().dot(X);
    linear_operator
        .solve_into(rhs)
        .or_else(|err| Err(format! {"{}", err}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    fn check_approx_eq(lhs: &Array1<f64>, rhs: &Array1<f64>) {
        let diff = lhs - rhs;
        let sq_diff = diff.dot(&diff);
        assert!(sq_diff < 1e-12);
    }

    #[test]
    fn fits_a_line_through_two_dots() {
        let lin_reg = LinearRegression::new().with_intercept();
        let A: Array2<f64> = array![[0.], [1.]];
        let b: Array1<f64> = array![1., 2.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let result = model.predict(&A);
        assert_eq!(result, array![1., 2.]);
    }

    /// When `with_intercept` is not set (the default), the
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
        let lin_reg = LinearRegression::new().with_intercept();
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
        let lin_reg = LinearRegression::new().with_intercept();
        let A: Array2<f64> = array![[0., 0.], [1., 1.], [2., 4.]];
        let b: Array1<f64> = array![1., 4., 9.];
        let model = lin_reg.fit(&A, &b).unwrap();
        let expected_params: Array1<f64> = array![1., 2., 1.];
        check_approx_eq(model.get_params(), &expected_params);
    }
}
