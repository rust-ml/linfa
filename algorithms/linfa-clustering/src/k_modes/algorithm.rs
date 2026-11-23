use std::collections::HashMap;

use crate::k_modes::init::KModesInit;
use crate::k_modes::{KModesError, KModesParams, KModesValidParams};
use linfa::traits::{Fit, PredictInplace};
use linfa::DatasetBase;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use ndarray_rand::rand::Rng;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq, Eq)]
/// K-Modes clustering model for categorical data (Huang \[1998\]).
///
/// Partitions $N$ categorical observations into $K$ clusters by finding a mode vector $Q_l$
/// for each cluster $l \in \{0, \dots, K-1\}$ that minimizes the sum of matching dissimilarities:
///
/// $$P(W, Q) = \sum_{l=0}^{K-1} \sum_{i=1}^{N} w_{il} \sum_{j=1}^{M} \delta(x_{ij}, q_{lj})$$
///
/// where:
/// - $w_{il} \in \{0, 1\}$ indicates whether observation $X_i$ belongs to cluster $l$,
/// - $\delta(x_{ij}, q_{lj}) = 0$ if $x_{ij} = q_{lj}$, and $1$ otherwise (Simple Matching Dissimilarity),
/// - $q_{lj}$ is the empirical mode (most frequent category) of attribute $j$ in cluster $l$.
///
/// ## Tutorial
///
/// ```
/// use linfa::DatasetBase;
/// use linfa::traits::{Fit, Predict};
/// use linfa_clustering::{KModes, KModesInit};
/// use ndarray::array;
///
/// // Create a categorical dataset
/// let entries = array![
///     ["Sunny", "Hot", "High"],
///     ["Sunny", "Hot", "High"],
///     ["Rainy", "Mild", "High"],
///     ["Rainy", "Cool", "Normal"],
///     ["Rainy", "Cool", "Normal"],
///     ["Sunny", "Hot", "High"]
/// ];
/// let dataset = DatasetBase::from(entries);
///
/// // Configure and fit K-Modes with 2 clusters
/// let model = KModes::params(2)
///     .max_n_iterations(100)
///     .init_method(KModesInit::Cao)
///     .fit(&dataset)
///     .expect("K-Modes fitting failed");
///
/// // Predict cluster assignments
/// let predictions = model.predict(&dataset);
/// assert_eq!(predictions.len(), 6);
///
/// // Query fitted cluster modes and total cost
/// assert_eq!(model.modes().dim(), (2, 3));
/// assert!(model.cost() <= 6);
/// ```
pub struct KModes<T: EquivalenceTarget> {
    modes: Array2<T>,
    cost: usize,
}

/// Trait bound for categorical elements supported by K-Modes.
pub trait EquivalenceTarget: PartialEq + Eq + Clone + std::hash::Hash {}
impl<T: PartialEq + Eq + Clone + std::hash::Hash> EquivalenceTarget for T {}

impl<T: EquivalenceTarget> KModes<T> {
    /// Constructs a parameter builder with `n_clusters` clusters and default RNG.
    pub fn params(n_clusters: usize) -> KModesParams<T, Xoshiro256Plus> {
        KModesParams::new(n_clusters)
    }

    /// Constructs a parameter builder with `n_clusters` clusters and a custom RNG.
    pub fn params_with_rng<R: Rng>(n_clusters: usize, rng: R) -> KModesParams<T, R> {
        KModesParams::new_with_rng(n_clusters, rng)
    }

    /// Cluster modes matrix with shape `(n_clusters, n_features)`.
    pub fn modes(&self) -> &Array2<T> {
        &self.modes
    }

    /// Total dissimilarity cost of the fitted clustering.
    pub fn cost(&self) -> usize {
        self.cost
    }
}

impl<T: EquivalenceTarget, R: Rng + Clone, D: Data<Elem = T>, L>
    Fit<ArrayBase<D, Ix2>, L, KModesError> for KModesValidParams<T, R>
{
    type Object = KModes<T>;

    /// Fits the K-Modes model on a categorical dataset.
    ///
    /// Executes `n_runs` independent clusterings (or 1 run for deterministic `Cao` initialization)
    /// and returns the model with minimal total dissimilarity cost.
    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, L>,
    ) -> Result<Self::Object, KModesError> {
        let observations = dataset.records().view();
        let (n_points, _) = observations.dim();

        if n_points == 0 {
            return Err(
                linfa::error::Error::Parameters("Dataset cannot be empty".to_string()).into(),
            );
        }

        if self.n_clusters() > n_points {
            return Err(linfa::error::Error::Parameters(format!(
                "Cannot have more clusters ({}) than data points ({})",
                self.n_clusters(),
                n_points
            ))
            .into());
        }

        let mut rng = self.rng().clone();
        let mut best_cost = usize::MAX;
        let mut best_centroids = None;

        let n_runs = match self.init_method() {
            KModesInit::Cao if self.n_runs() > 1 => {
                if self.verbose() {
                    println!("Cao initialization is deterministic. Running 1 initialization.");
                }
                1
            }
            KModesInit::Precomputed(_) if self.n_runs() > 1 => 1,
            _ => self.n_runs(),
        };

        for run_idx in 0..n_runs {
            if self.verbose() && n_runs > 1 {
                println!("Starting K-Modes run {}/{}", run_idx + 1, n_runs);
            }

            let (centroids, cost) = k_modes_single(
                observations,
                self.n_clusters(),
                self.max_n_iterations(),
                self.init_method(),
                &mut rng,
                self.verbose(),
            );

            if cost < best_cost {
                best_cost = cost;
                best_centroids = Some(centroids);
            }
        }

        let modes = best_centroids.expect("internal error: K-Modes failed to fit centroids");

        Ok(KModes {
            modes,
            cost: best_cost,
        })
    }
}

/// Simple matching dissimilarity: counts mismatched features between two categorical vectors.
///
/// $$d(X, Y) = \sum_{j=1}^{M} [x_j \neq y_j]$$
pub(crate) fn matching_dissim<T: PartialEq>(a: ArrayView1<T>, b: ArrayView1<T>) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// Finds the nearest centroid index and minimum distance for a given observation point.
pub(crate) fn closest_centroid<T: PartialEq>(
    centroids: ArrayView2<T>,
    observation: ArrayView1<T>,
) -> (usize, usize) {
    let mut min_dist = usize::MAX;
    let mut closest_idx = 0;

    for (idx, centroid) in centroids.rows().into_iter().enumerate() {
        let dist = matching_dissim(centroid, observation);
        if dist < min_dist {
            min_dist = dist;
            closest_idx = idx;
        }
    }

    (closest_idx, min_dist)
}

/// Evaluates total clustering loss: sum of matching distances of all observations to their assigned modes.
fn calculate_cost<T: PartialEq>(
    observations: ArrayView2<T>,
    centroids: ArrayView2<T>,
    memberships: &[usize],
) -> usize {
    observations
        .rows()
        .into_iter()
        .zip(memberships.iter())
        .map(|(obs, &cluster_id)| matching_dissim(obs, centroids.row(cluster_id)))
        .sum()
}

/// Single run of Huang's incremental online K-Modes algorithm.
///
/// Steps:
/// 1. Initialize $K$ modes using the configured initialization strategy (`Cao`, `Huang`, `Random`, `Precomputed`).
/// 2. Assign each observation to its closest initial mode and build frequency tables:
///    `cl_attr_freq[cluster][attribute][value] -> count`
/// 3. Recompute initial cluster modes by taking the argmax frequency per attribute.
/// 4. Iteratively cycle through all points:
///    - Determine the closest cluster mode for point $X_i$.
///    - If closest cluster $\neq$ current cluster, immediately move $X_i$, updating the category
///      frequency counts and modes for both the source and target clusters in $O(M)$ time.
///    - If the source cluster becomes empty, reinitialize it with a point drawn from the largest cluster.
/// 5. Repeat until no points change cluster membership (`moves == 0`) or `max_n_iterations` is reached.
fn k_modes_single<T: EquivalenceTarget, R: Rng>(
    observations: ArrayView2<T>,
    n_clusters: usize,
    max_n_iterations: u64,
    init: &KModesInit<T>,
    rng: &mut R,
    verbose: bool,
) -> (Array2<T>, usize) {
    let (n_points, n_attrs) = observations.dim();

    // centroid generation
    let mut centroids = init.run(n_clusters, observations, rng);
    let mut memberships = vec![0usize; n_points];
    let mut cluster_counts = vec![0usize; n_clusters];

    // Track category frequencies per cluster and feature: cl_attr_freq[k][j][v] -> count
    let mut cl_attr_freq: Vec<Vec<HashMap<T, usize>>> = (0..n_clusters)
        .map(|_| (0..n_attrs).map(|_| HashMap::new()).collect())
        .collect();

    for (ipoint, curpoint) in observations.rows().into_iter().enumerate() {
        let (clust, _) = closest_centroid(centroids.view(), curpoint);
        memberships[ipoint] = clust;
        cluster_counts[clust] += 1;

        for (iattr, curattr) in curpoint.iter().enumerate() {
            *cl_attr_freq[clust][iattr]
                .entry(curattr.clone())
                .or_insert(0) += 1;
        }
    }

    // Step 3: Compute initial empirical modes from frequency distributions
    for ik in 0..n_clusters {
        if cluster_counts[ik] == 0 {
            let random_idx = rng.gen_range(0..n_points);
            let sample_row = observations.row(random_idx);
            for iattr in 0..n_attrs {
                centroids[[ik, iattr]] = sample_row[iattr].clone();
            }
        } else {
            for iattr in 0..n_attrs {
                let cur_val = &centroids[[ik, iattr]];
                let cur_count = cl_attr_freq[ik][iattr].get(cur_val).copied().unwrap_or(0);
                if let Some((best_val, &max_count)) = cl_attr_freq[ik][iattr]
                    .iter()
                    .filter(|(_, &count)| count > 0)
                    .max_by_key(|(_, &count)| count)
                {
                    if max_count > cur_count {
                        centroids[[ik, iattr]] = best_val.clone();
                    }
                }
            }
        }
    }

    let mut cost = calculate_cost(observations, centroids.view(), &memberships);

    // Online incremental reassignment loop
    let mut iter = 0;
    while iter < max_n_iterations {
        iter += 1;
        let mut moves = 0;

        for ipoint in 0..n_points {
            let curpoint = observations.row(ipoint);
            let (to_clust, _) = closest_centroid(centroids.view(), curpoint);
            let from_clust = memberships[ipoint];

            if to_clust == from_clust {
                continue;
            }

            // Move point to closer cluster
            moves += 1;
            memberships[ipoint] = to_clust;
            cluster_counts[from_clust] -= 1;
            cluster_counts[to_clust] += 1;

            // Dynamically update frequency maps and centroid modes
            move_point_cat(
                curpoint,
                to_clust,
                from_clust,
                &mut cl_attr_freq,
                &mut centroids,
            );

            // Reassign a point from the largest cluster if a cluster became empty
            if cluster_counts[from_clust] == 0 {
                let largest_clust = cluster_counts
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &count)| count)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let candidates: Vec<usize> = memberships
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &c)| if c == largest_clust { Some(idx) } else { None })
                    .collect();

                if !candidates.is_empty() {
                    let rindx = candidates[rng.gen_range(0..candidates.len())];
                    let rpoint = observations.row(rindx);

                    memberships[rindx] = from_clust;
                    cluster_counts[largest_clust] -= 1;
                    cluster_counts[from_clust] += 1;

                    move_point_cat(
                        rpoint,
                        from_clust,
                        largest_clust,
                        &mut cl_attr_freq,
                        &mut centroids,
                    );
                }
            }
        }

        let ncost = calculate_cost(observations, centroids.view(), &memberships);
        cost = ncost;

        // Convergence check: loop terminates when no cluster reassignments occur during the epoch
        if moves == 0 {
            if verbose {
                println!(
                    "K-Modes converged at iteration {} (moves: {}, cost: {})",
                    iter, moves, cost
                );
            }
            break;
        } else if verbose && (iter % 10 == 0 || iter == max_n_iterations) {
            println!(
                "Iteration {}/{}: moves = {}, cost = {}",
                iter, max_n_iterations, moves, cost
            );
        }
    }

    (centroids, cost)
}

/// Dynamically updates attribute frequencies and modes when transferring a single observation between clusters.
///
/// - For `to_clust`: increments category frequency; if new frequency exceeds the current mode's count,
///   the centroid coordinate is updated to this category in $O(1)$ time.
/// - For `from_clust`: decrements category frequency; if the removed category was the current mode,
///   the new mode is recalculated by finding the key with maximum count.
fn move_point_cat<T: EquivalenceTarget>(
    point: ArrayView1<T>,
    to_clust: usize,
    from_clust: usize,
    cl_attr_freq: &mut [Vec<HashMap<T, usize>>],
    centroids: &mut Array2<T>,
) {
    for (iattr, curattr) in point.iter().enumerate() {
        // Increment frequency in destination cluster
        let to_map = &mut cl_attr_freq[to_clust][iattr];
        let to_count = to_map.entry(curattr.clone()).or_insert(0);
        *to_count += 1;
        let new_val_freq = *to_count;

        let current_centroid_val = &centroids[[to_clust, iattr]];
        let current_centroid_freq = to_map.get(current_centroid_val).copied().unwrap_or(0);
        if new_val_freq > current_centroid_freq {
            centroids[[to_clust, iattr]] = curattr.clone();
        }

        // Decrement frequency in source cluster
        let from_map = &mut cl_attr_freq[from_clust][iattr];
        if let Some(from_count) = from_map.get_mut(curattr) {
            if *from_count > 0 {
                *from_count -= 1;
            }
        }

        // If the departed value was the mode, check if another value is now strictly more frequent
        let old_centroid_val = &centroids[[from_clust, iattr]];
        if old_centroid_val == curattr {
            let remaining_count = from_map.get(curattr).copied().unwrap_or(0);
            if let Some((best_val, &max_count)) = from_map
                .iter()
                .filter(|(_, &count)| count > 0)
                .max_by_key(|(_, &count)| count)
            {
                if max_count > remaining_count {
                    centroids[[from_clust, iattr]] = best_val.clone();
                }
            }
        }
    }
}

impl<T: EquivalenceTarget, D: Data<Elem = T>> PredictInplace<ArrayBase<D, Ix2>, Array1<usize>>
    for KModes<T>
{
    /// Predicts closest cluster indices for a 2D batch of observations.
    fn predict_inplace(&self, observations: &ArrayBase<D, Ix2>, memberships: &mut Array1<usize>) {
        assert_eq!(observations.nrows(), memberships.len());

        for (i, obs) in observations.rows().into_iter().enumerate() {
            memberships[i] = closest_centroid(self.modes.view(), obs).0;
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<usize> {
        Array1::zeros(x.nrows())
    }
}

impl<T: EquivalenceTarget, D: Data<Elem = T>> PredictInplace<ArrayBase<D, Ix1>, usize>
    for KModes<T>
{
    /// Predicts closest cluster index for a single 1D observation.
    fn predict_inplace(&self, observation: &ArrayBase<D, Ix1>, membership: &mut usize) {
        *membership = closest_centroid(self.modes.view(), observation.view()).0;
    }

    fn default_target(&self, _x: &ArrayBase<D, Ix1>) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::traits::Predict;
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<KModes<usize>>();
        has_autotraits::<KModes<&str>>();
        has_autotraits::<KModes<String>>();
        has_autotraits::<KModes<u8>>();
        has_autotraits::<KModes<char>>();
        has_autotraits::<KModes<bool>>();
    }

    #[test]
    fn test_kmodes_simple_fitting() {
        let data = array![
            ["A", "X"],
            ["A", "X"],
            ["B", "Y"],
            ["B", "Y"],
            ["A", "X"],
            ["B", "Y"]
        ];

        let dataset = DatasetBase::from(data);
        let model = KModes::params(2)
            .max_n_iterations(50)
            .verbose(false)
            .fit(&dataset)
            .unwrap();

        assert_eq!(model.modes().dim(), (2, 2));

        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), 6);

        assert_eq!(predictions[0], predictions[1]);
        assert_eq!(predictions[0], predictions[4]);

        assert_eq!(predictions[2], predictions[3]);
        assert_eq!(predictions[2], predictions[5]);
        assert_ne!(predictions[0], predictions[2]);
        assert_eq!(model.cost(), 0);
    }

    #[test]
    fn test_kmodes_single_observation_predict() {
        let data = array![["A", "X"], ["B", "Y"]];
        let dataset = DatasetBase::from(data);
        let model = KModes::params(2).fit(&dataset).unwrap();

        let single_sample = array!["A", "X"];
        let pred: usize = model.predict(&single_sample);
        assert!(pred < 2);
    }

    #[test]
    fn test_kmodes_empty_dataset_error() {
        let empty_data: Array2<usize> = Array2::zeros((0, 3));
        let dataset = DatasetBase::from(empty_data);
        let result = KModes::params(2).fit(&dataset);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmodes_n_clusters_exceeds_samples_error() {
        let data = array![["A", "X"], ["B", "Y"]];
        let dataset = DatasetBase::from(data);
        let result = KModes::params(5).fit(&dataset);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmodes_single_cluster() {
        let data = array![["A", "X"], ["A", "X"], ["A", "Y"], ["B", "X"]];
        let dataset = DatasetBase::from(data);
        let model = KModes::params(1).fit(&dataset).unwrap();

        assert_eq!(model.modes().dim(), (1, 2));
        assert_eq!(model.modes()[[0, 0]], "A");
        assert_eq!(model.modes()[[0, 1]], "X");

        let predictions = model.predict(&dataset);
        assert_eq!(predictions.to_vec(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_kmodes_n_clusters_equals_n_samples() {
        let data = array![["A", "X"], ["B", "Y"], ["C", "Z"]];
        let dataset = DatasetBase::from(data.clone());
        let model = KModes::params(3)
            .init_method(KModesInit::Random)
            .fit(&dataset)
            .unwrap();

        assert_eq!(model.modes().dim(), (3, 2));
        assert_eq!(model.cost(), 0);

        let predictions = model.predict(&dataset);
        let mut unique_labels: Vec<usize> = predictions.to_vec();
        unique_labels.sort_unstable();
        unique_labels.dedup();
        assert_eq!(unique_labels.len(), 3);
    }

    #[test]
    fn test_kmodes_identical_observations() {
        let data = array![
            ["A", "X", "1"],
            ["A", "X", "1"],
            ["A", "X", "1"],
            ["A", "X", "1"]
        ];
        let dataset = DatasetBase::from(data);
        let model = KModes::params(2).fit(&dataset).unwrap();

        assert_eq!(model.cost(), 0);
        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_kmodes_high_dimensional_dataset() {
        let row_a = vec!["A"; 20];
        let row_b = vec!["B"; 20];
        let mut raw_data = Vec::new();
        for _ in 0..10 {
            raw_data.extend(row_a.clone());
        }
        for _ in 0..10 {
            raw_data.extend(row_b.clone());
        }
        let data = Array2::from_shape_vec((20, 20), raw_data).unwrap();
        let dataset = DatasetBase::from(data);

        let model = KModes::params(2)
            .max_n_iterations(20)
            .fit(&dataset)
            .unwrap();

        assert_eq!(model.modes().dim(), (2, 20));
        assert_eq!(model.cost(), 0);

        let preds = model.predict(&dataset);
        assert_eq!(preds.slice(ndarray::s![0..10]).to_vec(), vec![preds[0]; 10]);
        assert_eq!(
            preds.slice(ndarray::s![10..20]).to_vec(),
            vec![preds[10]; 10]
        );
        assert_ne!(preds[0], preds[10]);
    }

    #[test]
    fn test_kmodes_max_iterations_bound() {
        let data = array![
            ["A", "1"],
            ["A", "2"],
            ["B", "1"],
            ["B", "2"],
            ["C", "1"],
            ["C", "2"]
        ];
        let dataset = DatasetBase::from(data);
        let model = KModes::params(3).max_n_iterations(1).fit(&dataset).unwrap();

        assert_eq!(model.modes().dim(), (3, 2));
    }

    #[test]
    fn test_kmodes_n_runs_selects_minimal_cost() {
        let data = array![
            ["A", "X", "1"],
            ["A", "X", "2"],
            ["B", "Y", "1"],
            ["B", "Y", "2"],
            ["C", "Z", "1"],
            ["C", "Z", "2"]
        ];
        let dataset = DatasetBase::from(data);
        let model_multi = KModes::params(3)
            .init_method(KModesInit::Random)
            .n_runs(10)
            .fit(&dataset)
            .unwrap();

        let model_single = KModes::params(3)
            .init_method(KModesInit::Random)
            .n_runs(1)
            .fit(&dataset)
            .unwrap();

        assert!(model_multi.cost() <= model_single.cost() || model_single.cost() == 0);
    }

    #[test]
    fn test_kmodes_with_custom_rng() {
        let data = array![
            ["A", "X"],
            ["A", "X"],
            ["B", "Y"],
            ["B", "Y"],
            ["C", "Z"],
            ["C", "Z"]
        ];
        let dataset = DatasetBase::from(data);

        let rng1 = Xoshiro256Plus::seed_from_u64(42);
        let rng2 = Xoshiro256Plus::seed_from_u64(42);

        let model1 = KModes::params_with_rng(3, rng1)
            .init_method(KModesInit::Random)
            .fit(&dataset)
            .unwrap();

        let model2 = KModes::params_with_rng(3, rng2)
            .init_method(KModesInit::Random)
            .fit(&dataset)
            .unwrap();

        assert_eq!(model1.modes(), model2.modes());
        assert_eq!(model1.cost(), model2.cost());
    }

    #[test]
    fn test_kmodes_predict_unseen_categories() {
        let data = array![["A", "X"], ["A", "X"], ["B", "Y"], ["B", "Y"]];
        let dataset = DatasetBase::from(data);
        let model = KModes::params(2).fit(&dataset).unwrap();

        // Sample with category "Z" never seen in training
        let unseen = array![["A", "Z"]];
        let pred: usize = model.predict(&unseen.row(0));
        let expected_cluster = model.predict(&array!["A", "X"]);
        assert_eq!(pred, expected_cluster);
    }

    #[test]
    fn test_kmodes_types_numeric() {
        let data = array![
            [1usize, 10usize],
            [1usize, 10usize],
            [2usize, 20usize],
            [2usize, 20usize]
        ];
        let dataset = DatasetBase::from(data);
        let model = KModes::params(2).fit(&dataset).unwrap();
        assert_eq!(model.modes().dim(), (2, 2));
        assert_eq!(model.cost(), 0);

        let u8_data = array![[1u8, 2u8], [1u8, 2u8], [3u8, 4u8], [3u8, 4u8]];
        let u8_dataset = DatasetBase::from(u8_data);
        let u8_model = KModes::params(2).fit(&u8_dataset).unwrap();
        assert_eq!(u8_model.modes().dim(), (2, 2));
        assert_eq!(u8_model.cost(), 0);

        let i32_data = array![
            [-10i32, 100i32],
            [-10i32, 100i32],
            [50i32, -200i32],
            [50i32, -200i32]
        ];
        let i32_dataset = DatasetBase::from(i32_data);
        let i32_model = KModes::params(2).fit(&i32_dataset).unwrap();
        assert_eq!(i32_model.modes().dim(), (2, 2));
        assert_eq!(i32_model.cost(), 0);
    }

    #[test]
    fn test_kmodes_types_char_and_bool() {
        let char_data = array![['a', 'x'], ['a', 'x'], ['b', 'y'], ['b', 'y']];
        let char_dataset = DatasetBase::from(char_data);
        let char_model = KModes::params(2).fit(&char_dataset).unwrap();
        assert_eq!(char_model.modes().dim(), (2, 2));
        assert_eq!(char_model.cost(), 0);

        let bool_data = array![[true, false], [true, false], [false, true], [false, true]];
        let bool_dataset = DatasetBase::from(bool_data);
        let bool_model = KModes::params(2).fit(&bool_dataset).unwrap();
        assert_eq!(bool_model.modes().dim(), (2, 2));
        assert_eq!(bool_model.cost(), 0);
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Weather {
        Sunny,
        Rainy,
        Overcast,
    }

    #[test]
    fn test_kmodes_types_custom_enum() {
        let data = array![
            [Weather::Sunny, Weather::Sunny],
            [Weather::Sunny, Weather::Sunny],
            [Weather::Rainy, Weather::Rainy],
            [Weather::Rainy, Weather::Rainy],
            [Weather::Overcast, Weather::Overcast],
            [Weather::Overcast, Weather::Overcast]
        ];
        let dataset = DatasetBase::from(data);
        let model = KModes::params(3)
            .init_method(KModesInit::Cao)
            .fit(&dataset)
            .unwrap();

        assert_eq!(model.modes().dim(), (3, 2));
        assert_eq!(model.cost(), 0);

        let preds = model.predict(&dataset);
        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[2], preds[3]);
        assert_eq!(preds[4], preds[5]);
    }

    #[test]
    fn test_kmodes_predict_dataset_and_records_consistency() {
        let data = array![
            ["Cat", "Small"],
            ["Dog", "Large"],
            ["Cat", "Small"],
            ["Dog", "Large"]
        ];
        let dataset = DatasetBase::from(data.clone());
        let model = KModes::params(2).fit(&dataset).unwrap();

        let dataset_preds = model.predict(&dataset);
        let records_preds = model.predict(&data);

        assert_eq!(dataset_preds, records_preds);
        for (i, row) in data.rows().into_iter().enumerate() {
            let single_pred: usize = model.predict(&row);
            assert_eq!(dataset_preds[i], single_pred);
        }
    }

    #[test]
    fn test_kmodes_all_init_strategies() {
        let data = array![
            ["A", "1"],
            ["A", "1"],
            ["B", "2"],
            ["B", "2"],
            ["C", "3"],
            ["C", "3"]
        ];
        let dataset = DatasetBase::from(data);

        let m_cao = KModes::params(3)
            .init_method(KModesInit::Cao)
            .fit(&dataset)
            .unwrap();
        assert_eq!(m_cao.modes().dim(), (3, 2));

        let m_huang = KModes::params(3)
            .init_method(KModesInit::Huang)
            .fit(&dataset)
            .unwrap();
        assert_eq!(m_huang.modes().dim(), (3, 2));

        let m_random = KModes::params(3)
            .init_method(KModesInit::Random)
            .fit(&dataset)
            .unwrap();
        assert_eq!(m_random.modes().dim(), (3, 2));

        let precomputed = array![["A", "1"], ["B", "2"], ["C", "3"]];
        let m_pre = KModes::params(3)
            .init_method(KModesInit::Precomputed(precomputed))
            .fit(&dataset)
            .unwrap();
        assert_eq!(m_pre.modes().dim(), (3, 2));
        assert_eq!(m_pre.cost(), 0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_kmodes_serde() {
        let data = array![
            ["A".to_string(), "X".to_string()],
            ["A".to_string(), "X".to_string()],
            ["B".to_string(), "Y".to_string()],
            ["B".to_string(), "Y".to_string()]
        ];
        let dataset = DatasetBase::from(data);
        let model = KModes::params(2).fit(&dataset).unwrap();

        let serialized = serde_json::to_string(&model).unwrap();
        let deserialized: KModes<String> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(model, deserialized);
        assert_eq!(model.modes(), deserialized.modes());
        assert_eq!(model.cost(), deserialized.cost());
    }

    #[test]
    fn test_kmodes_verbose_logging_coverage() {
        let data = array![
            ["A", "X"],
            ["A", "X"],
            ["B", "Y"],
            ["B", "Y"],
            ["C", "Z"],
            ["C", "Z"]
        ];
        let dataset = DatasetBase::from(data);

        // Triggers Cao deterministic warning logging
        let m_cao = KModes::params(2)
            .n_runs(3)
            .verbose(true)
            .init_method(KModesInit::Cao)
            .fit(&dataset)
            .unwrap();
        assert_eq!(m_cao.modes().dim(), (2, 2));

        // Triggers multi-run and convergence verbose logging
        let m_rand = KModes::params(2)
            .n_runs(2)
            .max_n_iterations(20)
            .verbose(true)
            .init_method(KModesInit::Random)
            .fit(&dataset)
            .unwrap();
        assert_eq!(m_rand.modes().dim(), (2, 2));

        // Triggers max_n_iterations logging branch
        let m_max_iter = KModes::params(2)
            .n_runs(1)
            .max_n_iterations(1)
            .verbose(true)
            .init_method(KModesInit::Random)
            .fit(&dataset)
            .unwrap();
        assert_eq!(m_max_iter.modes().dim(), (2, 2));
    }

    #[test]
    fn test_kmodes_empty_cluster_recovery() {
        let data = array![
            ["A", "1"],
            ["A", "1"],
            ["A", "1"],
            ["A", "1"],
            ["B", "2"],
            ["B", "2"],
            ["B", "2"],
            ["B", "2"]
        ];
        let dataset = DatasetBase::from(data);

        // Precompute centroids with duplicate points so cluster 1 becomes empty during assignment
        let init_modes = array![["A", "1"], ["A", "1"], ["B", "2"]];
        let model = KModes::params(3)
            .init_method(KModesInit::Precomputed(init_modes))
            .fit(&dataset)
            .unwrap();

        assert_eq!(model.modes().dim(), (3, 2));
    }

    #[test]
    fn test_kmodes_default_target() {
        use linfa::traits::PredictInplace;
        let data = array![["A", "X"], ["B", "Y"]];
        let dataset = DatasetBase::from(data.clone());
        let model = KModes::params(2).fit(&dataset).unwrap();

        let t2d = model.default_target(&data);
        assert_eq!(t2d.len(), 2);

        let sample = array!["A", "X"];
        let t1d = model.default_target(&sample);
        assert_eq!(t1d, 0);
    }

    #[test]
    fn test_kmodes_initial_empirical_mode_update() {
        // Points assigned to cluster 0 will have "Y" as mode, differing from initial seed "X"
        let data = array![
            ["Y", "1"],
            ["Y", "1"],
            ["Y", "1"],
            ["X", "1"],
            ["Z", "9"],
            ["Z", "9"]
        ];
        let dataset = DatasetBase::from(data);

        // Precomputed initial centroids: cluster 0 starts with ["X", "1"], cluster 1 starts with ["Z", "9"]
        let init_modes = array![["X", "1"], ["Z", "9"]];
        let model = KModes::params(2)
            .init_method(KModesInit::Precomputed(init_modes))
            .fit(&dataset)
            .unwrap();

        assert_eq!(model.modes()[[0, 0]], "Y");
    }

    #[test]
    #[should_panic]
    fn test_kmodes_predict_inplace_mismatched_length() {
        use linfa::traits::PredictInplace;
        let data = array![["A", "1"], ["B", "2"]];
        let dataset = DatasetBase::from(data.clone());
        let model = KModes::params(2).fit(&dataset).unwrap();
        let mut wrong_memberships = ndarray::Array1::zeros(10);
        model.predict_inplace(&data, &mut wrong_memberships);
    }
}
