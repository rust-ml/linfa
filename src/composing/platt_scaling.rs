//! Implement Platt calibration with Newton method
//!

use crate::Float;
use crate::dataset::AsTargets;
use crate::traits::IncrementalFit;

use ndarray::{ArrayBase, Ix2, Data};

pub struct Platt<F> {
    maxiter: usize,
    minstep: F,
    eps: F,
}

impl<F: Float> Platt<F> {
    pub fn params() -> Self {
        Platt {
            maxiter: 100,
            minstep: F::from(1e-10).unwrap(),
            eps: F::from(1e-12).unwrap()
        }
    }
}

impl<'a, F: Float, D, T> IncrementalFit<'a, ArrayBase<D, Ix2>, T> for Platt<F>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = bool>,
{
}
