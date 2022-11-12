//! `linfa-clustering` aims to provide pure Rust implementations
//! of popular clustering algorithms.
//!
//! ## The big picture
//!
//! `linfa-clustering` is a crate in the `linfa` ecosystem, a wider effort to
//! bootstrap a toolkit for classical Machine Learning implemented in pure Rust,
//! kin in spirit to Python's `scikit-learn`.
//!
//! You can find a roadmap (and a selection of good first issues)
//! [here](https://github.com/LukeMathWalker/linfa/issues) - contributors are more than welcome!
//!
//! ## Current state
//!
//! Right now `linfa-clustering` provides the following clustering algorithms:
//! * [K-Means](KMeans)
//! * [DBSCAN](Dbscan)
//! * [Approximated DBSCAN](AppxDbscan)
//! * [Gaussian-Mixture-Model](GaussianMixtureModel)
//! * [OPTICS](OpticsAnalysis)
//!
//! Implementation choices, algorithmic details and tutorials can be found in the page dedicated to the specific algorithms.
mod appx_dbscan;
mod dbscan;
mod gaussian_mixture;
#[allow(clippy::new_ret_no_self)]
mod k_means;
mod optics;

pub use appx_dbscan::*;
pub use dbscan::*;
pub use gaussian_mixture::*;
pub use k_means::*;
pub use optics::*;
