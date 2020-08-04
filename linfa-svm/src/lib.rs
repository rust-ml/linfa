//! Support Vector Machines
//!
use std::fmt;
use ndarray::Array1;
use linfa_kernel::Kernel;

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
    exit_reason: ExitReason
}

impl SvmResult {
    pub fn classify(&self, kernel: &Kernel<f64>, data: Array1<f64>) -> f64 {
        let sum = kernel.weighted_sum(&self.alpha, data.view());

        sum - self.rho
    }
}

impl fmt::Display for SvmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.exit_reason {
            ExitReason::ReachedThreshold(obj, iter) => 
                write!(f, "Exited after {} iterations with obj = {}", iter, obj),
            ExitReason::ReachedIterations(obj, iter) => 
                write!(f, "Reached maximal iterations {} with obj = {}", iter, obj)
        }
    }
}


