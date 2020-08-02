//! Support Vector Machines
//!

pub mod solver;

pub struct SvmResult {
    alpha: Vec<f64>,
    rho: f64,
    obj: f64,
}
