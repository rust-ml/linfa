//! Support Vector Machines
//!
//! Support Vector Machines are supervised learning models and offer classification or
//! regression analysis for large problems. Normally they find a linear projection for the
//! task, but with the [kernel method](https://en.wikipedia.org/wiki/Kernel_method) non-linear relations between the input features
//! can be learned to improve the performance of the model.
//!
//! More details can be found [here](https://en.wikipedia.org/wiki/Support_vector_machine)
//!
//! # The solver
//! This implementation uses Sequential Minimal Optimization, a widely used optimization tool for
//! convex problems. It selects in each optimization step two variables and solves the sub-problem
//! in a closed form. After a couple of iterations the problem may converge.
//!
//! # Example
//! The wine quality data consists of 11 features, like "acid", "sugar", "sulfur dioxide", and
//! groups the quality into worst 3 to best 8. These are unified to good 8-7 and bad 3-6 to get a
//! binary classification task. 
//!
//! With an RBF kernel and C-Support Vector Classification an
//! accuracy of 0.988% is reached within 2911 iterations and 1248 support vectors.
//! ```
//! Fit SVM classifier with #1439 training points
//! Exited after 2911 iterations with obj = -248.51510322468084 and 1248 support vectors
//!
//! classes    | bad        | good
//! bad        | 1228       | 17
//! good       | 0          | 194
//!
//! accuracy 0.98818624, MCC 0.9523008
//! ```
use linfa_kernel::Kernel;
use ndarray::{ArrayBase, Data, Ix1, NdFloat};
use std::fmt;

mod classification;
mod permutable_kernel;
mod regression;
pub mod solver_smo;

pub use solver_smo::SolverParams;

/// Support Vector Classification
#[allow(non_snake_case)]
pub mod SVClassify {
    pub use crate::classification::{fit_c, fit_nu, fit_one_class};
}

/// Support Vector Regression
#[allow(non_snake_case)]
pub mod SVRegress {
    pub use crate::regression::{fit_epsilon, fit_nu};
}

/// An extension of the NdArray float type
pub trait Float: NdFloat + Default + Clone + std::iter::Sum {}

impl Float for f32 {}
impl Float for f64 {}

/// SMO can either exit because a threshold is reached or the iterations are maxed out
#[derive(Debug)]
pub enum ExitReason {
    ReachedThreshold,
    ReachedIterations,
}

/// The result of the SMO solver
pub struct SvmResult<'a, A: Float> {
    pub alpha: Vec<A>,
    pub rho: A,
    r: Option<A>,
    exit_reason: ExitReason,
    iterations: usize,
    obj: A,
    kernel: &'a Kernel<'a, A>,
}

impl<'a, A: Float> SvmResult<'a, A> {
    /// Predict new values with the model
    ///
    /// In case of a classification task this returns a probability, for regression the predicted
    /// regressor is returned. 
    pub fn predict<S: Data<Elem = A>>(&self, data: ArrayBase<S, Ix1>) -> A {
        let sum = self.kernel.weighted_sum(&self.alpha, data.view());

        sum - self.rho
    }

    /// Returns the number of support vectors
    pub fn nsupport(&self) -> usize {
        self.alpha
            .iter()
            .filter(|x| x.abs() > A::from(1e-5).unwrap())
            .count()
    }
}

impl<'a, A: Float> fmt::Display for SvmResult<'a, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.exit_reason {
            ExitReason::ReachedThreshold => write!(
                f,
                "Exited after {} iterations with obj = {} and {} support vectors",
                self.iterations,
                self.obj,
                self.nsupport()
            ),
            ExitReason::ReachedIterations => write!(
                f,
                "Reached maximal iterations {} with obj = {} and {} support vectors",
                self.iterations,
                self.obj,
                self.nsupport()
            ),
        }
    }
}
