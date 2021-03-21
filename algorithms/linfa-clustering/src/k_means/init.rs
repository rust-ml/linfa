use super::algorithm::closest_centroid;
use linfa::Float;
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

#[cfg(test)]
mod tests {
    use super::super::algorithm::{compute_inertia, update_cluster_memberships};
    use super::*;
    use approx::{abs_diff_eq, assert_abs_diff_eq, assert_abs_diff_ne};
    use ndarray::{array, stack, Array};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use rand_isaac::Isaac64Rng;
    use std::collections::HashSet;

    #[test]
    fn test_min_dists() {
        let centroids = array![[0.0, 1.0], [40.0, 10.0]];
        let observations = array![[3.0, 4.0], [1.0, 3.0], [25.0, 15.0]];
        let mut dists = Array1::zeros(observations.nrows());
        update_min_dists(&centroids, &observations, &mut dists);
        assert_abs_diff_eq!(dists, array![18.0, 5.0, 250.0]);
    }

    #[test]
    fn test_kmeans_pp() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let centroids = [20.0, -1000.0, 1000.0];
        let clusters: Vec<Array2<_>> = centroids
            .iter()
            .map(|&c| Array::random_using((50, 2), Normal::new(c, 1.).unwrap(), &mut rng))
            .collect();
        let obs = clusters.iter().fold(Array2::default((0, 2)), |a, b| {
            stack(Axis(0), &[a.view(), b.view()]).unwrap()
        });

        let out = k_means_pp(3, &obs.view(), &mut rng.clone());
        let mut cluster_ids = HashSet::new();
        for row in out.genrows() {
            // Centroid should not be 0
            assert_abs_diff_ne!(row, Array1::zeros(row.len()), epsilon = 1e-1);
            // Find the resultant centroid in 1 of the 3 clusters
            let found = clusters
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if c.genrows().into_iter().any(|cl| abs_diff_eq!(row, cl)) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .next()
                .unwrap();
            cluster_ids.insert(found);
        }
        // Centroids should almost always span all 3 clusters
        assert_eq!(cluster_ids, [0, 1, 2].iter().copied().collect());
    }

    #[test]
    fn test_compare() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let centroids = [20.0, -1000.0, 1000.0];
        let clusters: Vec<Array2<_>> = centroids
            .iter()
            .map(|&c| Array::random_using((50, 2), Normal::new(c, 1.).unwrap(), &mut rng))
            .collect();
        let obs = clusters.iter().fold(Array2::default((0, 2)), |a, b| {
            stack(Axis(0), &[a.view(), b.view()]).unwrap()
        });

        let out_rand = random_init(3, &obs.view(), &mut rng.clone());
        let out_pp = k_means_pp(3, &obs.view(), &mut rng.clone());
        // Inertia of Kmeans++ should be better than using random_init
        assert!(calc_inertia(&out_pp, &obs) < calc_inertia(&out_rand, &obs));
    }

    fn calc_inertia(
        centroids: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        observations: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    ) -> f64 {
        let mut memberships = Array1::zeros(observations.nrows());
        update_cluster_memberships(centroids, observations, &mut memberships);
        compute_inertia(centroids, observations, &memberships)
    }
}
