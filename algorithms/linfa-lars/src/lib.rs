//! # Least angle regression a.k.a. LAR
//!
//! This struct contains the parameters of a fitted LARS model. This includes the seperating
//! hyperplane, (optionally) intercept, alphas (Maximum of covariances (in absolute value) at each iteration),
//! Indices of active variables at the end of the path,
//!
//! LARS is similar to forward stepwise regression.
//! At each step, it finds the feature most correlated with the target.
//! When there are multiple features having equal correlation, instead of continuing along the same feature,
//! it proceeds in a direction equiangular between the features.
//!
//! ## References
//!
//! * ["Least Angle Regression", Efron et al.](https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)
//! * [Wikipedia entry on the Least-angle regression](https://en.wikipedia.org/wiki/Least-angle_regression)
//! * [Scikit-Learn User Guide](https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression)

use linfa::{Float, traits::PredictInplace};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

pub use error::LarsError;
pub use hyperparams::{LarsParams, LarsValidParams};

mod algorithm;
mod error;
mod hyperparams;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct Lars<F> {
    hyperplane: Array1<F>,
    intercept: F,
    alphas: Array1<F>,
    n_iter: usize,
    active: Vec<usize>,
    coef_path: Array2<F>,
}

impl<F: Float> Lars<F> {
    /// Create default Lars hyper parameters
    ///
    /// By default, an intercept will be fitted. To disable fitting an
    /// intercept, call `.fit_intercept(false)` before calling `.fit()`.
    ///
    /// The feature matrix will not be normalized by default.
    pub fn params() -> LarsParams<F> {
        LarsParams::new()
    }

    /// Get the varying values of the coefficients along the path.
    pub fn coef_path(&self) -> &Array2<F> {
        &self.coef_path
    }

    /// Get the fitted hyperplane
    pub fn hyperplane(&self) -> &Array1<F> {
        &self.hyperplane
    }

    /// Maximum of covariances (in absolute value) at each iteration
    pub fn alphas(&self) -> &Array1<F> {
        &self.alphas
    }

    /// The number of iterations taken by the algorithm to find the grid of alphas for each target
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Indices of active variables at the end of the path
    pub fn active(&self) -> &Vec<usize> {
        &self.active
    }

    /// Get the fitted intercept, 0. if no intercept was fitted
    pub fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<F>> for Lars<F> {
    /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
    /// `predict` returns the target variable according to LARS
    /// learned from the training data distribution.
    fn predict_inplace<'a>(&'a self, x: &'a ArrayBase<D, Ix2>, y: &mut Array1<F>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        *y = x.dot(&self.hyperplane) + self.intercept;
    }
    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        Array1::zeros(x.nrows())
    }
}
