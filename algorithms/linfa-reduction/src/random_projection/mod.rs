//! # Random Projections
//!
//! These algorithms build a low-distortion embedding of the input data
//! in a low-dimensional Euclidean space by projecting the data onto a random subspace.
//! The embedding is a randomly chosen matrix (either Gaussian or sparse),
//! following the [Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).
//!
//! This result states that, if the dimension of the embedding is `Ω(log(n_samples)/eps^2)`,
//! then with high probability, the projection `p` has distortion less than `eps`,
//! where `eps` is parameter, with `0 < eps < 1`.
//! "Distortion less than `eps`" means that for all vectors `u, v` in the original dataset,
//! we have `(1 - eps) d(u, v) <= d(p(u), p(v)) <= (1 + eps) d(u, v)`,
//! where `d` denotes the distance between two vectors.
//!
//! Note that the dimension of the embedding does not
//! depend on the original dimension of the data set (the number of features).
//!
//! ## Comparison with other methods
//!
//! To obtain a given accuracy on a given task, random projections will
//! often require a larger embedding dimension than other reduction methods such as PCA.
//! However, random projections have a very low computational cost,
//! since they only consist in sampling a random matrix,
//! whereas the PCA requires computing the pseudoinverse of a large matrix,
//! which is computationally expensive.
pub(crate) mod common;
pub(crate) mod gaussian;
pub(crate) mod projection;
pub(crate) mod sparse;

pub use gaussian::{
    GaussianRandomProjection, GaussianRandomProjectionParams, GaussianRandomProjectionValidParams,
};

pub use sparse::{
    SparseRandomProjection, SparseRandomProjectionParams, SparseRandomProjectionValidParams,
};

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;

    #[test]
    fn autotraits_gaussian() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<GaussianRandomProjection<f64>>();
        has_autotraits::<GaussianRandomProjection<f32>>();
        has_autotraits::<GaussianRandomProjectionValidParams<SmallRng>>();
        has_autotraits::<GaussianRandomProjectionParams<SmallRng>>();
    }

    #[test]
    fn autotraits_sparse() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<SparseRandomProjection<f64>>();
        has_autotraits::<SparseRandomProjection<f32>>();
        has_autotraits::<SparseRandomProjectionValidParams>();
        has_autotraits::<SparseRandomProjectionParams>();
    }
}
