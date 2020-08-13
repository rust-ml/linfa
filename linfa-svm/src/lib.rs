//! Support Vector Machines
//!
use linfa_kernel::Kernel;
use ndarray::{Array1, ArrayBase, Data, Ix1};
use std::fmt;

mod permutable_kernel;
pub mod hyperparameters;
pub mod solver_smo;

pub use hyperparameters::SolverParams;
pub use solver_smo::Classification;

#[derive(Debug)]
pub enum ExitReason {
    ReachedThreshold(f64, usize),
    ReachedIterations(f64, usize),
}

pub struct SvmResult<'a> {
    alpha: Vec<f64>,
    rho: f64,
    exit_reason: ExitReason,
    kernel: &'a Kernel<f64>
}

impl<'a> SvmResult<'a> {
    pub fn predict<S: Data<Elem = f64>>(
        &self,
        data: ArrayBase<S, Ix1>,
    ) -> f64 {
        let sum = self.kernel.weighted_sum(&self.alpha, data.view());

        sum - self.rho
    }

    pub fn nsupport(&self) -> usize {
        self.alpha.iter().filter(|x| x.abs() > 1e-5).count()
    }
}

impl<'a> fmt::Display for SvmResult<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.exit_reason {
            ExitReason::ReachedThreshold(obj, iter) => {
                write!(f, "Exited after {} iterations with obj = {} and {} support vectors", iter, obj, self.nsupport())
            }
            ExitReason::ReachedIterations(obj, iter) => {
                write!(f, "Reached maximal iterations {} with obj = {} and {} support vectors", iter, obj, self.nsupport())
            }
        }
    }
}
