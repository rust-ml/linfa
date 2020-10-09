use argmin::prelude::{ArgminAdd, ArgminDot, ArgminFloat, ArgminMul, ArgminNorm, ArgminSub};
use ndarray::{Array1, NdFloat};
use ndarray_linalg::Lapack;
use num_traits::float::FloatConst;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};

// A Float trait that captures the requirements we need for the various places
// we need floats. There requirements are imposed y ndarray and argmin
pub trait Float:
    ArgminFloat
    + FloatConst
    + NdFloat
    + Lapack
    + Default
    + Clone
    + FromPrimitive
    + ArgminMul<ArgminParam<Self>, ArgminParam<Self>>
{
    const POSITIVE_LABEL: Self;
    const NEGATIVE_LABEL: Self;
}

impl Float for f32 {
    const POSITIVE_LABEL: Self = 1.0;
    const NEGATIVE_LABEL: Self = -1.0;
}

impl Float for f64 {
    const POSITIVE_LABEL: Self = 1.0;
    const NEGATIVE_LABEL: Self = -1.0;
}

impl ArgminMul<ArgminParam<Self>, ArgminParam<Self>> for f64 {
    fn mul(&self, other: &ArgminParam<Self>) -> ArgminParam<Self> {
        ArgminParam(&other.0 * *self)
    }
}

impl ArgminMul<ArgminParam<Self>, ArgminParam<Self>> for f32 {
    fn mul(&self, other: &ArgminParam<Self>) -> ArgminParam<Self> {
        ArgminParam(&other.0 * *self)
    }
}

// Here we create a new type over ndarray's Array1. This is required
// to implement traits required by argmin
#[derive(Serialize, Clone, Deserialize, Debug, Default)]
pub struct ArgminParam<A>(pub Array1<A>);

impl<A> ArgminParam<A> {
    #[inline]
    pub fn as_array(&self) -> &Array1<A> {
        &self.0
    }
}

impl<A: Float> ArgminSub<ArgminParam<A>, ArgminParam<A>> for ArgminParam<A> {
    fn sub(&self, other: &ArgminParam<A>) -> ArgminParam<A> {
        ArgminParam(&self.0 - &other.0)
    }
}

impl<A: Float> ArgminAdd<ArgminParam<A>, ArgminParam<A>> for ArgminParam<A> {
    fn add(&self, other: &ArgminParam<A>) -> ArgminParam<A> {
        ArgminParam(&self.0 + &other.0)
    }
}

impl<A: Float> ArgminDot<ArgminParam<A>, A> for ArgminParam<A> {
    fn dot(&self, other: &ArgminParam<A>) -> A {
        self.0.dot(&other.0)
    }
}

impl<A: Float> ArgminNorm<A> for ArgminParam<A> {
    fn norm(&self) -> A {
        self.0.dot(&self.0)
    }
}

impl<A: Float> ArgminMul<A, ArgminParam<A>> for ArgminParam<A> {
    fn mul(&self, other: &A) -> ArgminParam<A> {
        ArgminParam(&self.0 * *other)
    }
}

impl<A: Float> ArgminMul<ArgminParam<A>, ArgminParam<A>> for ArgminParam<A> {
    fn mul(&self, other: &ArgminParam<A>) -> ArgminParam<A> {
        ArgminParam(&self.0 * &other.0)
    }
}
