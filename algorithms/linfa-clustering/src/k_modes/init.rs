use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand::Rng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
/// Specifies the centroid initialization strategy for K-Modes.
pub enum KModesInit<T> {
    /// Density and dissimilarity-based initialization (Cao et al. \[2009\]). Default.
    #[default]
    Cao,
    /// Density-based initialization (Huang \[1997, 1998\]).
    Huang,
    /// Randomly selects `n_clusters` unique observations from the dataset.
    Random,
    /// Precomputed initial centroids with shape `(n_clusters, n_features)`.
    Precomputed(Array2<T>),
}

impl<T: crate::k_modes::EquivalenceTarget> KModesInit<T> {
    /// Executes the configured initialization routine and returns initial centroids.
    pub(crate) fn run<R: Rng>(
        &self,
        n_clusters: usize,
        observations: ArrayView2<T>,
        rng: &mut R,
    ) -> Array2<T> {
        match self {
            Self::Cao => init_cao(observations, n_clusters),
            Self::Huang => init_huang(observations, n_clusters, rng),
            Self::Random => random_init(observations, n_clusters, rng),
            Self::Precomputed(centroids) => {
                assert_eq!(centroids.nrows(), n_clusters);
                assert_eq!(centroids.ncols(), observations.ncols());
                centroids.clone()
            }
        }
    }
}

/// Cao initialization (Cao et al. \[2009\]): Density and dissimilarity-based selection.
///
/// Steps:
/// 1. Compute marginal attribute density for every point $X_i$:
///    $$\text{Dens}(X_i) = \sum_{j=1}^{M} \frac{\text{freq}_j(x_{ij})}{N \cdot M}$$
///    where $\text{freq}_j(v)$ is the frequency of category $v$ across feature $j$,
///    $N$ is the sample count, and $M$ is the feature count.
/// 2. Pick the point with maximum density as the first centroid:
///    $$C_0 = \text{argmax}_{i} \text{Dens}(X_i)$$
/// 3. For subsequent centroids $k \in \{1, \dots, K-1\}$, choose the unselected point
///    that maximizes the minimum density-weighted distance to existing centroids:
///    $$C_k = \text{argmax}_{i \notin C} \left( \min_{c \in C} d(X_i, c) \cdot \text{Dens}(X_i) \right)$$
///    This balances selecting high-density centers while maintaining spatial dispersion.
pub(crate) fn init_cao<T: crate::k_modes::EquivalenceTarget>(
    x: ArrayView2<T>,
    n_clusters: usize,
) -> Array2<T> {
    let (n_points, n_attrs) = x.dim();
    assert!(n_clusters <= n_points);

    // Compute marginal attribute density per point
    let mut dens = vec![0.0f64; n_points];
    for iattr in 0..n_attrs {
        let mut freq: HashMap<&T, usize> = HashMap::new();
        for ipoint in 0..n_points {
            *freq.entry(&x[[ipoint, iattr]]).or_insert(0) += 1;
        }
        let denom = (n_points * n_attrs) as f64;
        for ipoint in 0..n_points {
            let count = freq.get(&x[[ipoint, iattr]]).copied().unwrap_or(0);
            dens[ipoint] += (count as f64) / denom;
        }
    }

    let mut selected_indices = Vec::with_capacity(n_clusters);

    // First centroid is the point with the highest density
    let first_idx = dens
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    selected_indices.push(first_idx);

    // Subsequent centroids maximize the minimum density-weighted distance to chosen centroids
    for _ in 1..n_clusters {
        let mut best_idx = 0;
        let mut best_score = -1.0f64;

        for (ipoint, &point_dens) in dens.iter().enumerate() {
            if selected_indices.contains(&ipoint) {
                continue;
            }

            let row = x.row(ipoint);
            let mut min_d = f64::INFINITY;

            for &c_idx in &selected_indices {
                let c_row = x.row(c_idx);
                let dist = row.iter().zip(c_row.iter()).filter(|(a, b)| a != b).count() as f64;
                let score = dist * point_dens;
                if score < min_d {
                    min_d = score;
                }
            }

            if min_d > best_score {
                best_score = min_d;
                best_idx = ipoint;
            }
        }
        selected_indices.push(best_idx);
    }

    Array2::from_shape_fn((n_clusters, n_attrs), |(i, j)| {
        x[[selected_indices[i], j]].clone()
    })
}

/// Huang initialization (Huang \[1997, 1998\]): Attribute frequency distribution sampling.
///
/// Steps:
/// 1. Construct $K$ tentative synthetic centroids where each attribute $j$ is independently
///    sampled proportional to its marginal category frequencies in the dataset.
/// 2. Because synthetic combinations may not represent real observations or could cause empty clusters,
///    each tentative centroid is mapped to its nearest distinct observation in $X$ via a linear scan.
pub(crate) fn init_huang<T: crate::k_modes::EquivalenceTarget, R: Rng>(
    x: ArrayView2<T>,
    n_clusters: usize,
    rng: &mut R,
) -> Array2<T> {
    let (n_points, n_attrs) = x.dim();
    assert!(n_clusters <= n_points);

    // Sample tentative centroids using column marginal distributions
    let mut tentative = Array2::from_shape_fn((n_clusters, n_attrs), |(_, j)| {
        let rand_idx = rng.gen_range(0..n_points);
        x[[rand_idx, j]].clone()
    });

    // Replace each tentative centroid with the closest unselected observation in X
    let mut selected_indices = HashSet::new();
    for ik in 0..n_clusters {
        let tent_row = tentative.row(ik);
        let mut best_idx = 0;
        let mut min_dist = usize::MAX;

        for ipoint in 0..n_points {
            if selected_indices.contains(&ipoint) {
                continue;
            }

            let row = x.row(ipoint);
            let dist = row
                .iter()
                .zip(tent_row.iter())
                .filter(|(a, b)| a != b)
                .count();

            if dist < min_dist {
                min_dist = dist;
                best_idx = ipoint;
            }
        }

        selected_indices.insert(best_idx);
        for j in 0..n_attrs {
            tentative[[ik, j]] = x[[best_idx, j]].clone();
        }
    }

    tentative
}

/// Random initialization: selects `n_clusters` distinct data points without replacement.
///
/// Uses an ordered vector with a hash set to guarantee seed-deterministic output ordering.
pub(crate) fn random_init<T: Clone, R: Rng>(
    x: ArrayView2<T>,
    n_clusters: usize,
    rng: &mut R,
) -> Array2<T> {
    let (n_points, n_attrs) = x.dim();
    assert!(n_clusters <= n_points);

    let mut selected_indices = Vec::with_capacity(n_clusters);
    let mut seen = HashSet::with_capacity(n_clusters);

    while selected_indices.len() < n_clusters {
        let idx = rng.gen_range(0..n_points);
        if seen.insert(idx) {
            selected_indices.push(idx);
        }
    }

    Array2::from_shape_fn((n_clusters, n_attrs), |(i, j)| {
        x[[selected_indices[i], j]].clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<KModesInit<usize>>();
    }

    #[test]
    fn test_init_cao_deterministic() {
        let data = array![
            ["A", "X", "1"],
            ["A", "X", "1"],
            ["A", "X", "2"],
            ["B", "Y", "2"],
            ["B", "Y", "2"],
            ["C", "Z", "3"]
        ];

        let c1 = KModesInit::Cao.run(2, data.view(), &mut Xoshiro256Plus::seed_from_u64(42));
        let c2 = KModesInit::Cao.run(2, data.view(), &mut Xoshiro256Plus::seed_from_u64(999));

        assert_eq!(c1.dim(), (2, 3));
        assert_eq!(c1, c2, "Cao initialization must be fully deterministic");

        // The first centroid in Cao is the highest density point ("A", "X", "2")
        let first_centroid = c1.row(0);
        assert_eq!(first_centroid[0], "A");
        assert_eq!(first_centroid[1], "X");
        assert_eq!(first_centroid[2], "2");
    }

    #[test]
    fn test_init_huang() {
        let data = array![
            ["A", "X"],
            ["A", "X"],
            ["B", "Y"],
            ["B", "Y"],
            ["C", "Z"],
            ["C", "Z"]
        ];

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let centroids = KModesInit::Huang.run(2, data.view(), &mut rng);

        assert_eq!(centroids.dim(), (2, 2));
        // Verify centroids are drawn from valid data points
        for row in centroids.rows() {
            let matches = data.rows().into_iter().any(|d_row| d_row == row);
            assert!(
                matches,
                "Huang centroid must match an existing data observation"
            );
        }
    }

    #[test]
    fn test_init_random_unique_indices() {
        let data = array![[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]];

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let centroids = KModesInit::Random.run(3, data.view(), &mut rng);

        assert_eq!(centroids.dim(), (3, 2));
        // Verify that 3 distinct rows from data were chosen
        let r0 = centroids.row(0);
        let r1 = centroids.row(1);
        let r2 = centroids.row(2);
        assert_ne!(r0, r1);
        assert_ne!(r0, r2);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_init_precomputed() {
        let data = array![["A", "X"], ["B", "Y"], ["C", "Z"]];
        let precomputed = array![["A", "X"], ["C", "Z"]];

        let init = KModesInit::Precomputed(precomputed.clone());
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let centroids = init.run(2, data.view(), &mut rng);

        assert_eq!(centroids, precomputed);
    }

    #[test]
    #[should_panic]
    fn test_init_precomputed_wrong_n_clusters() {
        let data = array![["A", "X"], ["B", "Y"], ["C", "Z"]];
        let precomputed = array![["A", "X"]];
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        KModesInit::Precomputed(precomputed).run(2, data.view(), &mut rng);
    }

    #[test]
    #[should_panic]
    fn test_init_precomputed_wrong_features() {
        let data = array![["A", "X"], ["B", "Y"], ["C", "Z"]];
        let precomputed = array![["A", "X", "extra"], ["B", "Y", "extra"]];
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        KModesInit::Precomputed(precomputed).run(2, data.view(), &mut rng);
    }

    #[test]
    fn test_init_single_cluster() {
        let data = array![["A", "X"], ["B", "Y"], ["A", "X"]];
        let mut rng = Xoshiro256Plus::seed_from_u64(42);

        let c_cao = KModesInit::Cao.run(1, data.view(), &mut rng);
        assert_eq!(c_cao.dim(), (1, 2));

        let c_huang = KModesInit::Huang.run(1, data.view(), &mut rng);
        assert_eq!(c_huang.dim(), (1, 2));

        let c_random = KModesInit::Random.run(1, data.view(), &mut rng);
        assert_eq!(c_random.dim(), (1, 2));
    }

    #[test]
    fn test_init_n_clusters_eq_n_points() {
        let data = array![["A", "X"], ["B", "Y"], ["C", "Z"]];
        let mut rng = Xoshiro256Plus::seed_from_u64(42);

        let c_cao = KModesInit::Cao.run(3, data.view(), &mut rng);
        assert_eq!(c_cao.dim(), (3, 2));

        let c_huang = KModesInit::Huang.run(3, data.view(), &mut rng);
        assert_eq!(c_huang.dim(), (3, 2));

        let c_random = KModesInit::Random.run(3, data.view(), &mut rng);
        assert_eq!(c_random.dim(), (3, 2));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_init() {
        let init: KModesInit<String> = KModesInit::Cao;
        let serialized = serde_json::to_string(&init).unwrap();
        let deserialized: KModesInit<String> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(init, deserialized);

        let precomputed = KModesInit::Precomputed(array![["A".to_string(), "B".to_string()]]);
        let serialized_pre = serde_json::to_string(&precomputed).unwrap();
        let deserialized_pre: KModesInit<String> = serde_json::from_str(&serialized_pre).unwrap();
        assert_eq!(precomputed, deserialized_pre);
    }

    #[test]
    #[should_panic]
    fn test_init_cao_n_clusters_exceeds_points() {
        let data = array![["A", "1"]];
        init_cao(data.view(), 5);
    }

    #[test]
    #[should_panic]
    fn test_init_huang_n_clusters_exceeds_points() {
        let data = array![["A", "1"]];
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        init_huang(data.view(), 5, &mut rng);
    }

    #[test]
    #[should_panic]
    fn test_init_random_n_clusters_exceeds_points() {
        let data = array![["A", "1"]];
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        random_init(data.view(), 5, &mut rng);
    }
}
