//! Dimensionality reduction techniques
//!
//! This crate provides algorithms for dimensionality reduction in data analysis. They can be used
//! to transform data from a high-dimensional space into a lower dimensional space such that some
//! property of the data is retained. 
//!
//! The following implementations are available:
//!  * Principal Component Analysis - projects data linearily and retains the largest variance
//!  * Diffusion Map - applies kernel methods and projects close regions together
//!
#[macro_use]
extern crate ndarray;

pub mod diffusion_map;
pub mod pca;
pub mod utils;

pub use diffusion_map::{DiffusionMap, DiffusionMapHyperParams};
pub use pca::Pca;
pub use utils::to_gaussian_similarity;
