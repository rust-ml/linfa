use argmin::prelude::ArgminFloat;
use ndarray::NdFloat;
use ndarray_linalg::Lapack;
use num_traits::FromPrimitive;

/// A Float trait that captures the requirements we need for the various
/// places we use floats. These are basically imposed by NdArray and Argmin.
pub trait Float: ArgminFloat + NdFloat + Lapack + Default + Clone + FromPrimitive {
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
