use linfa::{traits::*, DatasetBase, Float};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, DataMut, Ix1, Ix2, Zip};
use ndarray_rand::rand;
use ndarray_rand::rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KMeansInit {
    Random,
}

impl KMeansInit {
    pub(crate) fn run<F: Float>(
        &self,
        n_clusters: usize,
        observations: &ArrayView2<F>,
        rng: &mut impl Rng,
    ) -> Array2<F> {
        match self {
            Self::Random => random_init(n_clusters, observations, rng),
        }
    }
}

fn random_init<F: Float>(
    n_clusters: usize,
    observations: &ArrayView2<F>,
    rng: &mut impl Rng,
) -> Array2<F> {
    let (n_samples, _) = observations.dim();
    let indices = rand::seq::index::sample(rng, n_samples, n_clusters).into_vec();
    observations.select(Axis(0), &indices)
}
