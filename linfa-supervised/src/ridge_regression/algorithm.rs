#![allow(non_snake_case)]
use ndarray::{Array, Array1, ArrayBase, Data, Ix1, Ix2};
use ndarray_linalg::Solve;
/* The difference between a linear regression and a Ridge regression is
 that ridge regression has an L2 penalisation term to having some features
 "taking all the credit" for the output. It is also a way to deal with over-fitting by adding bias.
 Some details ...
 b = (X^T X + aI)X^T y with a being the regularisation/penalisation term
*/

pub struct RidgeRegression {
    beta: Option<Array1<f64>>,
    alpha: f64,
}

impl RidgeRegression {
    pub fn new(alpha: f64) -> RidgeRegression {
        RidgeRegression {
            beta: None,
            alpha: alpha,
        }
    }

    pub fn fit<A, B>(&mut self, X: &ArrayBase<A, Ix2>, Y: &ArrayBase<B, Ix1>)
    where
        A: Data<Elem = f64>,
        B: Data<Elem = f64>,
    {
        let second_term = X.t().dot(Y);
        let (_, identity_size) = X.dim();
        let linear_operator = X.t().dot(X) + self.alpha * Array::eye(identity_size);
        self.beta = Some(linear_operator.solve_into(second_term).unwrap());
    }

    pub fn predict<A>(&self, X: &ArrayBase<A, Ix2>) -> Array1<f64>
    where
        A: Data<Elem = f64>,
    {
        match &self.beta {
            None => panic!("The ridge regression estimator has to be fitted first!"),
            Some(beta) => X.dot(beta),
        }
    }
}
