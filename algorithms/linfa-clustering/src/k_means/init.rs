use super::algorithm::{closest_centroid, update_cluster_memberships};
use linfa::Float;
use ndarray::parallel::prelude::*;
use ndarray::{
    s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, DataMut, Ix1, Ix2, Zip,
};
use ndarray_rand::rand;
use ndarray_rand::rand::distributions::{uniform::SampleUniform, Distribution, WeightedIndex};
use ndarray_rand::rand::Rng;
use std::ops::AddAssign;

pub type RngFunc<F> = fn(u64) -> F;

#[derive(Clone, Debug, PartialEq)]
pub enum KMeansInit<F: Float + SampleUniform + for<'a> AddAssign<&'a F>> {
    Random,
    KMeansPlusPlus,
    KMeansPara(RngFunc<F>),
}

impl<'a, F: Float + SampleUniform + for<'b> AddAssign<&'b F>> KMeansInit<F> {
    pub(crate) fn run(
        &self,
        n_clusters: usize,
        observations: &'a ArrayView2<'a, F>,
        rng: &mut impl Rng,
    ) -> Array2<F> {
        match self {
            Self::Random => random_init(n_clusters, observations, rng),
            Self::KMeansPlusPlus => k_means_plusplus(n_clusters, observations, rng),
            Self::KMeansPara(rng_func) => k_means_para(n_clusters, observations, rng, rng_func),
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

fn weighted_k_means_plusplus<F: Float + SampleUniform + for<'a> AddAssign<&'a F>>(
    n_clusters: usize,
    observations: &ArrayView2<F>,
    weights: &ArrayView1<F>,
    rng: &mut impl Rng,
) -> Array2<F> {
    let (n_samples, n_features) = observations.dim();
    assert_eq!(n_samples, weights.len());

    let mut centroids = Array2::zeros((n_clusters, n_features));
    let first_idx = WeightedIndex::new(weights.iter())
        .expect("invalid weights")
        .sample(rng);
    centroids.row_mut(0).assign(&observations.row(first_idx));

    let mut dists = Array1::zeros(n_samples);
    for c_cnt in 1..n_clusters {
        update_min_dists(&centroids.slice(s![0..c_cnt, ..]), observations, &mut dists);
        dists *= weights;
        let centroid_idx = WeightedIndex::new(dists.iter())
            .expect("invalid weights")
            .sample(rng);
        centroids
            .row_mut(c_cnt)
            .assign(&observations.row(centroid_idx));
    }
    centroids
}

fn k_means_plusplus<F: Float + SampleUniform + for<'a> AddAssign<&'a F>>(
    n_clusters: usize,
    observations: &ArrayView2<F>,
    rng: &mut impl Rng,
) -> Array2<F> {
    weighted_k_means_plusplus(
        n_clusters,
        observations,
        &Array1::ones(observations.nrows()).view(),
        rng,
    )
}

fn k_means_para<'a, F: Float + SampleUniform + for<'b> AddAssign<&'b F>>(
    n_clusters: usize,
    observations: &'a ArrayView2<'a, F>,
    rng: &mut impl Rng,
    rng_func: &RngFunc<F>,
) -> Array2<F> {
    let n_rounds = 8;
    let candidates_per_round = n_clusters;

    let (n_samples, n_features) = observations.dim();
    let mut candidates = Array2::zeros((n_clusters * n_rounds, n_features));

    let first_idx = rng.gen_range(0, n_samples);
    candidates.row_mut(0).assign(&observations.row(first_idx));
    let mut n_candidates = 1;

    let mut dists = Array1::zeros(n_samples);
    'outer: for _ in 0..n_rounds {
        let current_candidates = candidates.slice(s![0..n_candidates, ..]);
        update_min_dists(&current_candidates, &observations, &mut dists);
        let next_candidates = sample_subsequent_candidates(
            observations,
            &dists,
            F::from(candidates_per_round).unwrap(),
            rng.gen_range(0, 100),
            rng_func,
        );

        for candidate in next_candidates.into_iter() {
            candidates.row_mut(n_candidates).assign(&candidate);
            n_candidates += 1;
            if n_candidates >= candidates.nrows() {
                break 'outer;
            }
        }
    }

    let final_candidates = candidates.slice(s![0..n_candidates, ..]);
    let weights = cluster_membership_counts(&final_candidates, &observations);
    weighted_k_means_plusplus(n_clusters, &final_candidates, &weights.view(), rng)
}

fn sample_subsequent_candidates<'a, F: Float>(
    observations: &'a ArrayView2<'a, F>,
    dists: &Array1<F>,
    candidates_per_round: F,
    seed: usize,
    rng_func: &RngFunc<F>,
) -> Vec<ArrayView1<'a, F>> {
    let cost = dists.sum();
    dists
        .axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .filter_map(|(i, d)| {
            let d = *d.into_scalar();
            let rand = rng_func(i as u64 * seed as u64);
            let prob = candidates_per_round * d / cost;
            if rand < prob {
                Some(observations.row(i))
            } else {
                None
            }
        })
        .collect()
}

fn cluster_membership_counts<F: Float>(
    centroids: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
    observations: &ArrayBase<impl Data<Elem = F> + Sync, Ix2>,
) -> Array1<F> {
    let n_samples = observations.nrows();
    let n_clusters = centroids.nrows();
    let mut memberships = Array1::zeros(n_samples);
    update_cluster_memberships(&centroids, observations, &mut memberships);
    let mut counts = Array1::zeros(n_clusters);
    memberships.iter().for_each(|&c| counts[c] += F::one());
    counts
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
    fn test_sample_subsequent_candidates() {
        fn func(_: u64) -> f64 {
            0.4
        }

        let observations = array![[3.0, 4.0], [1.0, 3.0], [25.0, 15.0]];
        let obs_view = observations.view();
        let dists = array![0.1, 0.4, 0.5];
        let candidates =
            sample_subsequent_candidates(&obs_view, &dists, 4.0, 0, &(func as RngFunc<f64>));
        assert_eq!(candidates.len(), 2);
        assert_abs_diff_eq!(candidates[0], observations.row(1));
        assert_abs_diff_eq!(candidates[1], observations.row(2));
    }

    #[test]
    fn test_cluster_membership_counts() {
        let centroids = array![[0.0, 1.0], [40.0, 10.0]];
        let observations = array![[3.0, 4.0], [1.0, 3.0], [25.0, 15.0]];
        let counts = cluster_membership_counts(&centroids, &observations);
        assert_abs_diff_eq!(counts, array![2.0, 1.0]);
    }

    #[test]
    fn test_weighted_kmeans_plusplus() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let obs = Array::random_using((1000, 2), Normal::new(0.0, 100.).unwrap(), &mut rng);
        let mut weights = Array1::zeros(1000);
        weights[0] = 1.0;
        weights[1] = 1.0;
        let out = weighted_k_means_plusplus(2, &obs.view(), &weights.view(), &mut rng);
        let mut expected_centroids = {
            let mut arr = Array2::zeros((2, 2));
            arr.row_mut(0).assign(&obs.row(0));
            arr.row_mut(1).assign(&obs.row(1));
            arr
        };
        assert!(
            abs_diff_eq!(out, expected_centroids) || {
                expected_centroids.invert_axis(Axis(0));
                abs_diff_eq!(out, expected_centroids)
            }
        );
    }

    #[test]
    fn test_k_means_plusplus() {
        verify_init(KMeansInit::KMeansPlusPlus);
    }

    #[test]
    fn test_k_means_para() {
        // Using Isaac here is quite slow but yields reproduceable results
        fn isaac(seed: u64) -> f64 {
            let mut rng = Isaac64Rng::seed_from_u64(seed);
            rng.gen_range(0.0, 1.0)
        }
        verify_init(KMeansInit::KMeansPara(isaac as RngFunc<f64>));
    }

    fn verify_init(init: KMeansInit<f64>) {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let centroids = [20.0, -1000.0, 1000.0];
        let clusters: Vec<Array2<_>> = centroids
            .iter()
            .map(|&c| Array::random_using((50, 2), Normal::new(c, 1.).unwrap(), &mut rng))
            .collect();
        let obs = clusters.iter().fold(Array2::default((0, 2)), |a, b| {
            stack(Axis(0), &[a.view(), b.view()]).unwrap()
        });

        let out = init.run(3, &obs.view(), &mut rng.clone());
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
        let out_pp = k_means_plusplus(3, &obs.view(), &mut rng.clone());
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
