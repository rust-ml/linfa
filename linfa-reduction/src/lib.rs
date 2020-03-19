#[macro_use] extern crate ndarray;

pub mod diffusion_map;
pub mod pca;
pub mod utils;

pub use pca::PrincipalComponentAnalysis;
pub use diffusion_map::{DiffusionMap, DiffusionMapHyperParams};
pub use utils::to_gaussian_similarity;

pub enum Method {
    DiffusionMap,
    PrincipalComponentAnalysis
}
