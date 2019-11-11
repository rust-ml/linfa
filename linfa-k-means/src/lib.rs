use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_rand::rand;
use ndarray_rand::rand::Rng;
use ndarray_stats::DeviationExt;
use std::collections::HashMap;

pub fn k_means(
    n_clusters: usize,
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    rng: &mut impl Rng,
    tolerance: f64,
    max_n_iterations: usize,
) -> Array2<f64> {
    let mut centroids = get_random_centroids(n_clusters, observations, rng);

    let mut has_converged;
    let mut n_iterations = 0;

    loop {
        let memberships = compute_cluster_memberships(&centroids, observations);
        let new_centroids = compute_centroids(observations, &memberships);

        let distance = centroids.sq_l2_dist(&new_centroids).unwrap();
        has_converged = distance < tolerance || n_iterations > max_n_iterations;

        centroids = new_centroids;
        n_iterations += 1;

        if has_converged {
            break;
        }
    }

    centroids
}

pub fn compute_centroids(
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    // (n_observations,)
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
) -> Array2<f64> {
    let centroids_hashmap = compute_centroids_hashmap(&observations, &cluster_memberships);

    // Go back to "cluster generation / dataset" if you are looking for inspiration!
    let n_centroids = centroids_hashmap.len();
    let (_, n_features) = observations.dim();

    let mut centroids: Array2<f64> = Array2::zeros((n_centroids, n_features));
    for (centroid_index, centroid) in centroids_hashmap.into_iter() {
        centroids
            .slice_mut(s![centroid_index, ..])
            .assign(&centroid.current_mean);
    }
    centroids
}

/// Iterate over our observations and capture in a HashMap the new centroids.
/// The HashMap is a (cluster_index => new centroid) mapping.
pub fn compute_centroids_hashmap(
    // (n_observations, n_features)
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    // (n_observations,)
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
) -> HashMap<usize, IncrementalMean> {
    let mut new_centroids: HashMap<usize, IncrementalMean> = HashMap::new();
    Zip::from(observations.genrows())
        .and(cluster_memberships)
        .apply(|observation, cluster_membership| {
            if let Some(incremental_mean) = new_centroids.get_mut(cluster_membership) {
                incremental_mean.update(&observation);
            } else {
                new_centroids.insert(
                    *cluster_membership,
                    IncrementalMean::new(observation.to_owned()),
                );
            }
        });
    new_centroids
}

pub struct IncrementalMean {
    pub current_mean: Array1<f64>,
    pub n_observations: usize,
}

impl IncrementalMean {
    pub fn new(first_observation: Array1<f64>) -> Self {
        Self {
            current_mean: first_observation,
            n_observations: 1,
        }
    }

    pub fn update(&mut self, new_observation: &ArrayBase<impl Data<Elem = f64>, Ix1>) {
        self.n_observations += 1;
        let shift =
            (new_observation - &self.current_mean).mapv_into(|x| x / self.n_observations as f64);
        self.current_mean += &shift;
    }
}

pub fn compute_cluster_memberships(
    centroids: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array1<usize> {
    // `map_axis` returns an array with one less dimension -
    // e.g. a 1-dimensional array if applied to a 2-dimensional array.
    //
    // Each 1-dimensional slice along the specified axis is replaced with the output value
    // of the closure passed as argument.
    observations.map_axis(Axis(1), |observation| {
        closest_centroid(&centroids, &observation)
    })
}

pub fn closest_centroid(
    centroids: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    observation: &ArrayBase<impl Data<Elem = f64>, Ix1>,
) -> usize {
    // Remember: you can use `.genrows().into_iter()` to get an iterator over the rows
    // of a 2-dimensional array.
    let mut iterator = centroids.genrows().into_iter().peekable();

    let first_centroid = iterator
        .peek()
        .expect("There has to be at least one centroid");
    let (mut closest_index, mut minimum_distance) =
        (0, first_centroid.sq_l2_dist(&observation).unwrap());

    for (centroid_index, centroid) in iterator.enumerate() {
        let distance = centroid.sq_l2_dist(&observation).unwrap();
        if distance < minimum_distance {
            closest_index = centroid_index;
            minimum_distance = distance;
        }
    }
    closest_index
}

pub fn get_random_centroids<S>(
    n_clusters: usize,
    observations: &ArrayBase<S, Ix2>,
    rng: &mut impl Rng,
) -> Array2<f64>
where
    // `Data` has an associated type, `Elem`, the element type.
    // This syntax tells the compiler that `Elem` is `f64`,
    // hence we are dealing with an array of floats.
    S: Data<Elem = f64>,
{
    let (n_samples, _) = observations.dim();
    let indices = rand::seq::index::sample(rng, n_samples, n_clusters).into_vec();
    observations.select(Axis(0), &indices)
}
