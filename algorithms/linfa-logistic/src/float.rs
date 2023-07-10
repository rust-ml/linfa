use crate::argmin_param::ArgminParam;
use argmin::core::ArgminFloat;
use argmin_math::ArgminMul;
use ndarray::{Dimension, Ix1, Ix2, NdFloat};
use num_traits::FromPrimitive;

/// A Float trait that captures the requirements we need for the various
/// places we use floats. These are basically imposed by NdArray and Argmin.
pub trait Float:
    ArgminFloat
    + NdFloat
    + Default
    + Clone
    + FromPrimitive
    + ArgminMul<ArgminParam<Self, Ix1>, ArgminParam<Self, Ix1>>
    + ArgminMul<ArgminParam<Self, Ix2>, ArgminParam<Self, Ix2>>
    + linfa::Float
{
    const POSITIVE_LABEL: Self;
    const NEGATIVE_LABEL: Self;
}

impl<D: Dimension> ArgminMul<ArgminParam<Self, D>, ArgminParam<Self, D>> for f64 {
    fn mul(&self, other: &ArgminParam<Self, D>) -> ArgminParam<Self, D> {
        ArgminParam(&other.0 * *self)
    }
}

impl<D: Dimension> ArgminMul<ArgminParam<Self, D>, ArgminParam<Self, D>> for f32 {
    fn mul(&self, other: &ArgminParam<Self, D>) -> ArgminParam<Self, D> {
        ArgminParam(&other.0 * *self)
    }
}

impl Float for f32 {
    const POSITIVE_LABEL: Self = 1.0;
    const NEGATIVE_LABEL: Self = -1.0;
}

impl Float for f64 {
    const POSITIVE_LABEL: Self = 1.0;
    const NEGATIVE_LABEL: Self = -1.0;
}
