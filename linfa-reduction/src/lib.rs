#[macro_use]
extern crate ndarray;

pub mod diffusion_map;
pub mod fast_ica;
pub mod pca;
pub mod utils;

pub use diffusion_map::{DiffusionMap, DiffusionMapHyperParams};
pub use pca::PrincipalComponentAnalysis;
pub use utils::to_gaussian_similarity;

use ndarray::NdFloat;
use ndarray_linalg::Lapack;
use num_traits::FromPrimitive;

pub trait Float: NdFloat + Lapack + Default + Clone + FromPrimitive {}

impl Float for f32 {}
impl Float for f64 {}
