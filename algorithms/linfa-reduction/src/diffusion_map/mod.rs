//! Diffusion Map
//!
//! The diffusion map computes an embedding of the data by applying PCA on the diffusion operator
//! of the data. It transforms the data along the direction of the largest diffusion flow and is therefore
//! a non-linear dimensionality reduction technique. A normalized kernel describes the high dimensional
//! diffusion graph with the (i, j) entry the probability that a diffusion happens from point i to
//! j.
//!
mod algorithms;
mod hyperparams;

pub use algorithms::*;
pub use hyperparams::*;
