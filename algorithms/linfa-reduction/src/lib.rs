#![doc = include_str!("../README.md")]

#[macro_use]
extern crate ndarray;

mod diffusion_map;
mod error;
mod pca;
pub mod utils;

pub use diffusion_map::{DiffusionMap, DiffusionMapParams, DiffusionMapValidParams};
pub use error::{ReductionError, Result};
pub use pca::{Pca, PcaParams};
pub use utils::to_gaussian_similarity;
