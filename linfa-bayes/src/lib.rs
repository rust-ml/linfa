mod error;
mod gaussian_nb;

use std::cmp::Ordering;

use ndarray::NdFloat;
use ndarray_linalg::Lapack;
use num_traits::FromPrimitive;

pub use gaussian_nb::GaussianNb;

pub trait Float:
    PartialEq + PartialOrd + NdFloat + Lapack + Default + Clone + FromPrimitive
{
}

impl Float for f32 {}
impl Float for f64 {}
