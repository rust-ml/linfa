//! This module defines newtypes for ndarray's Array1.
//!
//! This is necessary to be able to abstract over floats (f32 and f64) so that
//! the logistic regression code can be abstract in the float type it works
//! with.
//!
//! Unfortunately, this requires that we re-implement some traits from Argmin.

use crate::float::Float;
use argmin::prelude::*;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Clone, Deserialize, Debug, Default)]
pub struct ArgminParam<F>(pub Array1<F>);

impl<F> ArgminParam<F> {
    #[inline]
    pub fn as_array(&self) -> &Array1<F> {
        &self.0
    }
}

impl<F: Float> ArgminSub<ArgminParam<F>, ArgminParam<F>> for ArgminParam<F> {
    fn sub(&self, other: &ArgminParam<F>) -> ArgminParam<F> {
        ArgminParam(&self.0 - &other.0)
    }
}

impl<F: Float> ArgminAdd<ArgminParam<F>, ArgminParam<F>> for ArgminParam<F> {
    fn add(&self, other: &ArgminParam<F>) -> ArgminParam<F> {
        ArgminParam(&self.0 + &other.0)
    }
}

impl<F: Float> ArgminDot<ArgminParam<F>, F> for ArgminParam<F> {
    fn dot(&self, other: &ArgminParam<F>) -> F {
        self.0.dot(&other.0)
    }
}

impl<F: Float> ArgminNorm<F> for ArgminParam<F> {
    fn norm(&self) -> F {
        self.0.dot(&self.0)
    }
}

impl<F: Float> ArgminMul<F, ArgminParam<F>> for ArgminParam<F> {
    fn mul(&self, other: &F) -> ArgminParam<F> {
        ArgminParam(&self.0 * *other)
    }
}

impl<F: Float> ArgminMul<ArgminParam<F>, ArgminParam<F>> for ArgminParam<F> {
    fn mul(&self, other: &ArgminParam<F>) -> ArgminParam<F> {
        ArgminParam(&self.0 * &other.0)
    }
}
