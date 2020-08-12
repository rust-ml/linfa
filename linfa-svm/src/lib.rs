//! Support Vector Machines
//!
use linfa_kernel::Kernel;
use ndarray::{Array1, ArrayBase, Data, Ix1};
use std::fmt;

pub mod hyperparameters;
pub mod solver_smo;

pub use hyperparameters::SolverParams;
pub use solver_smo::Classification;

#[derive(Debug)]
pub enum ExitReason {
    ReachedThreshold(f64, usize),
    ReachedIterations(f64, usize),
}

#[derive(Debug)]
pub struct SvmResult {
    alpha: Vec<f64>,
    rho: f64,
    exit_reason: ExitReason,
}

impl SvmResult {
    pub fn predict<S: Data<Elem = f64>>(
        &self,
        kernel: &Kernel<f64>,
        data: ArrayBase<S, Ix1>,
    ) -> f64 {
        let sum = kernel.weighted_sum(&self.alpha, data.view());

        sum - self.rho
    }
}

impl fmt::Display for SvmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.exit_reason {
            ExitReason::ReachedThreshold(obj, iter) => {
                write!(f, "Exited after {} iterations with obj = {}", iter, obj)
            }
            ExitReason::ReachedIterations(obj, iter) => {
                write!(f, "Reached maximal iterations {} with obj = {}", iter, obj)
            }
        }
    }
}
