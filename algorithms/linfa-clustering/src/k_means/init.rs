use super::algorithm::{update_cluster_memberships, update_min_dists};
use linfa::Float;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_rand::rand::distributions::{uniform::SampleUniform, Distribution, WeightedIndex};
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::{self, SeedableRng};
use std::{
    ops::AddAssign,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
/// Specifies centroid initialization algorithm for KMeans.
pub enum KMeansInit {
    /// Pick random points as centroids.
    Random,
    /// K-means++ algorithm. Using this over random initialization causes K-means to converge
    /// faster for almost all cases, since K-means++ produces better centroids.
    KMeansPlusPlus,
    /// K-means|| algorithm, a parallelized version of K-means++. Performs much better than
    /// K-means++ when the number of clusters is large (>100) while producing similar centroids, so
    /// use this for larger datasets.  Details on the algorithm can be found
    /// [here](http://vldb.org/pvldb/vol5/p622_bahmanbahmani_vldb2012.pdf).
    KMeansPara,
}

impl KMeansInit {
    /// Runs the chosen initialization routine
    pub(crate) fn run<R: Rng + SeedableRng, F: Float + SampleUniform + for<'b> AddAssign<&'b F>>(
        &self,
        n_clusters: usize,
        observations: ArrayView2<F>,
        rng: &mut R,
    ) -> Array2<F> {
        match self {
            Self::Random => random_init(n_clusters, observations, rng),
            Self::KMeansPlusPlus => k_means_plusplus(n_clusters, observations, rng),
            Self::KMeansPara => k_means_para(n_clusters, observations, rng),
        }
    }
}

/// Pick random points from the input matrix as centroids
fn random_init<F: Float>(
    n_clusters: usize,
    observations: ArrayView2<F>,
    rng: &mut impl Rng,
) -> Array2<F> {
    let (n_samples, _) = observations.dim();
    let indices = rand::seq::index::sample(rng, n_samples, n_clusters).into_vec();
    observations.select(Axis(0), &indices)
}

/// Selects centroids using the KMeans++ initialization algorithm. The weights determine the
/// likeliness of an input point to be selected as a centroid relative to other points. The higher
/// the weight, the more likely the point will be selected as a centroid.
fn weighted_k_means_plusplus<F: Float + SampleUniform + for<'a> AddAssign<&'a F>>(
    n_clusters: usize,
    observations: ArrayView2<F>,
    weights: ArrayView1<F>,
    rng: &mut impl Rng,
) -> Array2<F> {
    let (n_samples, n_features) = observations.dim();
    assert_eq!(n_samples, weights.len());
    assert_ne!(weights.sum(), F::zero());

    let mut centroids = Array2::zeros((n_clusters, n_features));
    // Select 1st centroid from the input randomly purely based on the weights.
    let first_idx = WeightedIndex::new(weights.iter())
        .expect("invalid weights")
        .sample(rng);
    centroids.row_mut(0).assign(&observations.row(first_idx));

    let mut dists = Array1::zeros(n_samples);
    for c_cnt in 1..n_clusters {
        update_min_dists(
            &centroids.slice(s![0..c_cnt, ..]),
            &observations,
            &mut dists,
        );

        // The probability of a point being selected as the next centroid is proportional to its
        // distance from its closest centroid multiplied by its weight.
        dists *= &weights;
        let centroid_idx = WeightedIndex::new(dists.iter())
            .map(|idx| idx.sample(rng))
            // This only errs if all of dists is 0, which means every point is assigned to a
            // centroid, so extra centroids don't matter and can be any index.
            .unwrap_or(0);
        centroids
            .row_mut(c_cnt)
            .assign(&observations.row(centroid_idx));
    }
    centroids
}

/// KMeans++ initialization algorithm without biased weights
fn k_means_plusplus<F: Float + SampleUniform + for<'a> AddAssign<&'a F>>(
    n_clusters: usize,
    observations: ArrayView2<F>,
    rng: &mut impl Rng,
) -> Array2<F> {
    weighted_k_means_plusplus(
        n_clusters,
        observations,
        Array1::ones(observations.nrows()).view(),
        rng,
    )
}

/// KMeans|| initialization algorithm
/// In each iteration, pick some new "candidate centroids" by sampling the probabilities of each
/// input point in parallel. The probability of a point becoming a centroid is the same as with
/// KMeans++. After multiple iterations, run weighted KMeans++ on the candidates to produce the
/// final set of centroids.
fn k_means_para<R: Rng + SeedableRng, F: Float + SampleUniform + for<'b> AddAssign<&'b F>>(
    n_clusters: usize,
    observations: ArrayView2<F>,
    rng: &mut R,
) -> Array2<F> {
    // The product of these parameters must exceed n_clusters. The higher they are, the more
    // candidates are selected, which improves the quality of the centroids but increases running
    // time. The values provided here are "sweetspots" suggested by the paper.
    let n_rounds = 8;
    let candidates_per_round = n_clusters;

    let (n_samples, n_features) = observations.dim();
    let mut candidates = Array2::zeros((n_clusters * n_rounds, n_features));

    // Pick 1st centroid randomly
    let first_idx = rng.gen_range(0, n_samples);
    candidates.row_mut(0).assign(&observations.row(first_idx));
    let mut n_candidates = 1;

    let mut dists = Array1::zeros(n_samples);
    'outer: for _ in 0..n_rounds {
        let current_candidates = candidates.slice(s![0..n_candidates, ..]);
        update_min_dists(&current_candidates, &observations, &mut dists);
        // Generate the next set of candidates from the input points, using the same probability
        // formula as KMeans++. On average this generates candidates equal to
        // `candidates_per_round`.
        let next_candidates_idx = sample_subsequent_candidates::<R, _>(
            &dists,
            F::from(candidates_per_round).unwrap(),
            rng.gen_range(0, std::u64::MAX),
        );

        // Append the newly generated candidates to the current cadidates, breaking out of the loop
        // if too many candidates have been found
        for idx in next_candidates_idx.into_iter() {
            candidates
                .row_mut(n_candidates)
                .assign(&observations.row(idx));
            n_candidates += 1;
            if n_candidates >= candidates.nrows() {
                break 'outer;
            }
        }
    }

    let final_candidates = candidates.slice(s![0..n_candidates, ..]);
    // Weigh the candidate centroids by the sizes of the clusters they form in the input points.
    let weights = cluster_membership_counts(&final_candidates, &observations);

    // The number of candidates is almost certainly higher than the number of centroids, so we
    // recluster the candidates into the right number of centroids using weighted KMeans++.
    weighted_k_means_plusplus(n_clusters, final_candidates, weights.view(), rng)
}

/// Generate candidate centroids by sampling each observation in parallel using a seedable RNG in
/// every thread. Average number of generated candidates should equal `multiplier`.
fn sample_subsequent_candidates<R: Rng + SeedableRng, F: Float>(
    dists: &Array1<F>,
    multiplier: F,
    seed: u64,
) -> Vec<usize> {
    // This sum can also be parallelized
    let cost = dists.sum();
    // Using an atomic allows the seed to be modified while seeding RNGs in parallel
    let seed = AtomicU64::new(seed);

    // Use `map_init` to generate an unique RNG for each Rayon thread, allowing both RNG creation
    // and random number generation to be parallelized. Alternative approaches included generating
    // an RNG for every observation and sequentially taking `multiplier` samples from a weighted
    // index of `dists`. Generating for every observation was too slow, and the sequential approach
    // yielded lower-quality centroids, so this approach was chosen. See PR #108 for more details.
    dists
        .axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .map_init(
            || R::seed_from_u64(seed.fetch_add(1, Relaxed)),
            move |rng, (i, d)| {
                let d = *d.into_scalar();
                let rand = F::from(rng.gen_range(0.0, 1.0)).unwrap();
                let prob = multiplier * d / cost;
                (i, rand, prob)
            },
        )
        .filter_map(|(i, rand, prob)| if rand < prob { Some(i) } else { None })
        .collect()
}

/// Returns the number of observation points that belong to each cluster.
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
        let observations = array![[3.0, 4.0], [1.0, 3.0], [25.0, 15.0]];
        let dists = array![0.1, 0.4, 0.5];
        let candidates = sample_subsequent_candidates::<Isaac64Rng, _>(&dists, 4.0, 0);
        assert_eq!(candidates.len(), 2);
        assert_abs_diff_eq!(observations.row(candidates[0]), observations.row(1));
        assert_abs_diff_eq!(observations.row(candidates[1]), observations.row(2));
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
        weights[0] = 2.0;
        weights[1] = 3.0;
        let out = weighted_k_means_plusplus(2, obs.view(), weights.view(), &mut rng);
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
        verify_init(KMeansInit::KMeansPara);
    }

    // Run general tests for a given init algorithm
    fn verify_init(init: KMeansInit) {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        // Make sure we don't panic on degenerate data (n_clusters > n_samples)
        let degenerate_data = array![[1.0, 2.0]];
        let out = init.run(2, degenerate_data.view(), &mut rng);
        assert_abs_diff_eq!(out, stack![Axis(0), degenerate_data, degenerate_data]);

        // Build 3 separated clusters of points
        let centroids = [20.0, -1000.0, 1000.0];
        let clusters: Vec<Array2<_>> = centroids
            .iter()
            .map(|&c| Array::random_using((50, 2), Normal::new(c, 1.).unwrap(), &mut rng))
            .collect();
        let obs = clusters.iter().fold(Array2::default((0, 2)), |a, b| {
            stack(Axis(0), &[a.view(), b.view()]).unwrap()
        });

        // Look for the right number of centroids
        let out = init.run(centroids.len(), obs.view(), &mut rng);
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

        let out_rand = random_init(3, obs.view(), &mut rng.clone());
        let out_pp = k_means_plusplus(3, obs.view(), &mut rng.clone());
        let out_para = k_means_para(3, obs.view(), &mut rng.clone());
        // Loss of Kmeans++ should be better than using random_init
        assert!(calc_loss(&out_pp, &obs) < calc_loss(&out_rand, &obs));
        // Loss of Kmeans|| should be better than using random_init
        assert!(calc_loss(&out_para, &obs) < calc_loss(&out_rand, &obs));
    }

    fn calc_loss(
        centroids: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        observations: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    ) -> f64 {
        let mut memberships = Array1::zeros(observations.nrows());
        update_cluster_memberships(centroids, observations, &mut memberships);
        compute_inertia(centroids, observations, &memberships)
    }
}
