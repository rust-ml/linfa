//! Support Vector Machines
//!
use linfa_kernel::Kernel;
use ndarray::{ArrayBase, Data, Ix1, NdFloat};
use std::fmt;

pub mod hyperparameters;
mod permutable_kernel;
pub mod solver_smo;

pub use hyperparameters::SolverParams;
pub use solver_smo::{Classification, Regression};

pub trait Float: NdFloat + Default + Clone + std::iter::Sum {}

impl Float for f32 {}
impl Float for f64 {}

#[derive(Debug)]
pub enum ExitReason {
    ReachedThreshold,
    ReachedIterations,
}

pub struct SvmResult<'a, A: Float> {
    alpha: Vec<A>,
    rho: A,
    r: Option<A>,
    exit_reason: ExitReason,
    iterations: usize,
    obj: A,
    kernel: &'a Kernel<A>,
}

impl<'a, A: Float> SvmResult<'a, A> {
    pub fn predict<S: Data<Elem = A>>(&self, data: ArrayBase<S, Ix1>) -> A {
        let sum = self.kernel.weighted_sum(&self.alpha, data.view());

        sum - self.rho
    }

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
