#![allow(non_snake_case)]
use ndarray::{stack, Array, Array1, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_linalg::Solve;
/* I will probably change the implementation for an enum for more type safety.
I have to make sure, it is a great idea when it comes to pyhton interoperability
enum Intercept {
    NoIntercept,
    Intercept(Array1<f64>)
}
pub struct LinearRegressor {
    beta : Option<Array1<f64>>,
    intercept : Intercept,
}
*/

/*
If fit_intercept is false, we suppose that the regression passes throught the origin
*/
/*
The simple linear regression model is
    y = bX + e  where e ~ N(0, sigma^2 * I)
In probabilistic terms this corresponds to
    y - bX ~ N(0, sigma^2 * I)
    y | X, b ~ N(bX, sigma^2 * I)
The loss for the model is simply the squared error between the model
predictions and the true values:
    Loss = ||y - bX||^2
The maximum likelihood estimation for the model parameters `beta` can be computed
in closed form via the normal equation:
    b = (X^T X)^{-1} X^T y
where (X^T X)^{-1} X^T is known as the pseudoinverse or Moore-Penrose inverse.
*/
pub struct LinearRegression {
    beta: Option<Array1<f64>>,
    fit_intercept: bool,
}

impl LinearRegression {
    pub fn new(fit_intercept: bool) -> LinearRegression {
        LinearRegression {
            beta: None,
            fit_intercept,
        }
    }
    /* Instead of assert_eq we should probably return a Result, we first have to have a generic error type for all algorithms */
    pub fn fit<A, B>(&mut self, X: &ArrayBase<A, Ix2>, Y: &ArrayBase<B, Ix1>)
    where
        A: Data<Elem = f64>,
        B: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();

        // We have to make sure that the dimensions match
        assert_eq!(Y.dim(), n_samples);

        self.beta = if self.fit_intercept {
            let dummy_column: Array<f64, _> = Array::ones((n_samples, 1));
            /*
                if x is has 2 features and 3 samples
                x = [[1,2]
                    ,[3,4]
                    ,[5,6]]
                dummy_column = [[1]
                               ,[1]
                               ,[1]]
            */
            let X_with_ones = stack(Axis(1), &[dummy_column.view(), X.view()]).unwrap();
            Some(LinearRegression::fit_beta(&X_with_ones, Y))
        } else {
            Some(LinearRegression::fit_beta(X, Y))
        }
    }
    fn fit_beta<A, B>(X: &ArrayBase<A, Ix2>, y: &ArrayBase<B, Ix1>) -> Array1<f64>
    where
        A: Data<Elem = f64>,
        B: Data<Elem = f64>,
    {
        let rhs = X.t().dot(y);
        let linear_operator = X.t().dot(X);
        linear_operator.solve_into(rhs).unwrap()
    }

    pub fn predict<A>(&self, X: &ArrayBase<A, Ix2>) -> Array1<f64>
    where
        A: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();

        // If we are fitting the intercept, we need an additional column
        if self.fit_intercept {
            let dummy_column: Array<f64, _> = Array::ones((n_samples, 1));
            let X = stack(Axis(1), &[dummy_column.view(), X.view()]).unwrap();
            match &self.beta {
                None => panic!("The linear regression estimator has to be fitted first!"),
                Some(beta) => X.dot(beta),
            }
        } else {
            match &self.beta {
                None => panic!("The linear regression estimator has to be fitted first!"),
                Some(beta) => X.dot(beta),
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;
    #[test]
    fn linear_regression_test() {
        let mut linear_regression = LinearRegression::new(false);
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        linear_regression.fit(&x, &y);
        let x_hat = array![[6.0]];
        assert_eq!(linear_regression.predict(&x_hat), array![6.0])
    }
}
