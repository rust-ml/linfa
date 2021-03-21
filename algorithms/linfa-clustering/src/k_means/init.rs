use super::algorithm::closest_centroid;
use linfa::{traits::*, DatasetBase, Float};
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, DataMut, Ix1, Ix2, Zip};
use ndarray_rand::rand;
use ndarray_rand::rand::distributions::{uniform::SampleUniform, Distribution, WeightedIndex};
use ndarray_rand::rand::Rng;
use std::ops::AddAssign;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KMeansInit {
    Random,
    KMeansPP,
}

impl KMeansInit {
    pub(crate) fn run<F: Float + SampleUniform + for<'a> AddAssign<&'a F>>(
        &self,
        n_clusters: usize,
        observations: &ArrayView2<F>,
        rng: &mut impl Rng,
    ) -> Array2<F> {
        match self {
            Self::Random => random_init(n_clusters, observations, rng),
            Self::KMeansPP => k_means_pp(n_clusters, observations, rng),
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

fn k_means_pp<F: Float + SampleUniform + for<'a> AddAssign<&'a F>>(
    n_clusters: usize,
    observations: &ArrayView2<F>,
    rng: &mut impl Rng,
) -> Array2<F> {
    let (n_samples, n_features) = observations.dim();
    let mut centroids = Array2::zeros((n_clusters, n_features));
    let n = rng.gen_range(0, n_samples);
    centroids.row_mut(0).assign(&observations.row(n));

    let mut dists = Array1::zeros(n_samples);
    for c_cnt in 1..n_clusters {
        update_min_dists(&centroids.slice(s![0..c_cnt, ..]), observations, &mut dists);
        let centroid_idx = WeightedIndex::new(dists.iter())
            .expect("invalid weights")
            .sample(rng);
        centroids
            .row_mut(c_cnt)
            .assign(&observations.row(centroid_idx));
    }
    centroids
}

fn update_min_dists<F: Float>(
    centroids: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    observations: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    dists: &mut ArrayBase<impl DataMut<Elem = F>, Ix1>,
) {
    Zip::from(observations.axis_iter(Axis(0)))
        .and(dists)
        .par_apply(|observation, dist| *dist = closest_centroid(&centroids, &observation).1);
}
