//! This module defines newtypes for ndarray's Array.
//!
//! This is necessary to be able to abstract over floats (f32 and f64) so that
//! the logistic regression code can be abstract in the float type it works
//! with.
//!
//! Unfortunately, this requires that we re-implement some traits from Argmin.

use crate::float::Float;
use argmin_math::{ArgminAdd, ArgminDot, ArgminL2Norm, ArgminMul, ArgminSub};
use ndarray::{Array, ArrayBase, Data, Dimension, Zip};

pub fn elem_dot<F: linfa::Float, A1: Data<Elem = F>, A2: Data<Elem = F>, D: Dimension>(
    a: &ArrayBase<A1, D>,
    b: &ArrayBase<A2, D>,
) -> F {
    Zip::from(a)
        .and(b)
        .fold(F::zero(), |acc, &a, &b| acc + a * b)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArgminParam<F, D: Dimension>(pub Array<F, D>);

impl<F, D: Dimension> ArgminParam<F, D> {
    #[inline]
    pub fn as_array(&self) -> &Array<F, D> {
        &self.0
    }
}

impl<F: Float, D: Dimension> ArgminSub<ArgminParam<F, D>, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn sub(&self, other: &ArgminParam<F, D>) -> ArgminParam<F, D> {
        ArgminParam(&self.0 - &other.0)
    }
}

impl<F: Float, D: Dimension> ArgminAdd<ArgminParam<F, D>, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn add(&self, other: &ArgminParam<F, D>) -> ArgminParam<F, D> {
        ArgminParam(&self.0 + &other.0)
    }
}

impl<F: Float, D: Dimension> ArgminDot<ArgminParam<F, D>, F> for ArgminParam<F, D> {
    fn dot(&self, other: &ArgminParam<F, D>) -> F {
        elem_dot(&self.0, &other.0)
    }
}

impl<F: Float, D: Dimension> ArgminL2Norm<F> for ArgminParam<F, D> {
    fn l2_norm(&self) -> F {
        num_traits::Float::sqrt(elem_dot(&self.0, &self.0))
    }
}

impl<F: Float, D: Dimension> ArgminMul<F, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn mul(&self, other: &F) -> ArgminParam<F, D> {
        ArgminParam(&self.0 * *other)
    }
}

impl<F: Float, D: Dimension> ArgminMul<ArgminParam<F, D>, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn mul(&self, other: &ArgminParam<F, D>) -> ArgminParam<F, D> {
        ArgminParam(&self.0 * &other.0)
    }
}
