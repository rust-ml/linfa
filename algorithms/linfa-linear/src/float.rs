use argmin::core::ArgminFloat;
use ndarray::NdFloat;
use num_traits::float::FloatConst;
use num_traits::FromPrimitive;

// A Float trait that captures the requirements we need for the various places
// we need floats. There requirements are imposed y ndarray and argmin
pub trait Float:
    ArgminFloat + FloatConst + NdFloat + Default + Clone + FromPrimitive + linfa::Float
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
