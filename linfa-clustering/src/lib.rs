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
//! [here](https://github.com/LukeMathWalker/linfa) - contributors are more than welcome!
//!
//! ## Current state
//!
//! Right now `linfa-clustering` only provides a single algorithm, `K-Means`, with
//! a couple of helper functions.
//!
//! Implementation choices, algorithmic details and a tutorial can be found [here](struct.KMeans.html).
//!
//! Check [here]() for extensive benchmarks against `scikit-learn`'s K-means implementation.

mod k_means;
mod utils;

pub use k_means::*;
pub use utils::*;
