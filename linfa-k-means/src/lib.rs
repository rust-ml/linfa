use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, DataMut, Ix1, Ix2, Zip};
use ndarray_rand::rand;
use ndarray_rand::rand::Rng;
use ndarray_stats::DeviationExt;
use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct KMeans {
    tolerance: f64,
    max_n_iterations: u64,
    // Only set after `fit` has been called
    centroids: Option<Array2<f64>>,
}

impl KMeans {
    pub fn new(tolerance: Option<f64>, max_n_iterations: Option<u64>) -> Self {
        Self {
            tolerance: tolerance.unwrap_or(1e-4),
            max_n_iterations: max_n_iterations.unwrap_or(300),
            centroids: None,
        }
    }

    // `observations`: (n_observations, n_features)
    pub fn fit(
        &mut self,
        n_clusters: usize,
        observations: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        rng: &mut impl Rng
    ) {
        let mut centroids = get_random_centroids(n_clusters, observations, rng);

        let mut has_converged;
        let mut n_iterations = 0;

        let mut memberships = Array1::zeros(observations.dim().0);

        loop {
            update_cluster_memberships(&centroids, observations, &mut memberships);
            let new_centroids = compute_centroids(n_clusters, observations, &memberships);

            let distance = centroids
                .sq_l2_dist(&new_centroids)
                .expect("Failed to compute distance");
            has_converged = distance < self.tolerance || n_iterations > self.max_n_iterations;

            centroids = new_centroids;
            n_iterations += 1;

            if has_converged {
                break;
            }
        }

        self.centroids = Some(centroids);
    }

    pub fn predict(&self, observations: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<usize> {
        compute_cluster_memberships(
            self.centroids
                .as_ref()
                .expect("The model has to be fitted before calling predict!"),
            observations,
        )
    }

    pub fn save(&self, path: PathBuf) -> std::io::Result<()> {
        std::fs::write(
            path,
           serde_json::to_string(&self)?
        )
    }

    pub fn load(path: PathBuf) -> Result<Self, anyhow::Error> {
        let reader = &std::fs::File::open(path)?;
        let model = serde_json::from_reader(reader)?;
        Ok(model)
    }

    pub fn centroids(&self) -> Option<&Array2<f64>> {
        self.centroids.as_ref()
    }
}

fn compute_centroids(
    n_clusters: usize,
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    cluster_memberships: &ArrayBase<impl Data<Elem = usize>, Ix1>,
) -> Array2<f64> {
    let centroids_hashmap = compute_centroids_hashmap(&observations, &cluster_memberships);
    let (_, n_features) = observations.dim();

    let mut centroids: Array2<f64> = Array2::zeros((n_clusters, n_features));
    for (centroid_index, centroid) in centroids_hashmap.into_iter() {
        centroids
            .slice_mut(s![centroid_index, ..])
            .assign(&centroid.current_mean);
    }
    centroids
}

/// Iterate over our observations and capture in a HashMap the new centroids.
/// The HashMap is a (cluster_index => new centroid) mapping.
fn compute_centroids_hashmap(
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

struct IncrementalMean {
    pub current_mean: Array1<f64>,
    pub n_observations: usize,
}

impl IncrementalMean {
    fn new(first_observation: Array1<f64>) -> Self {
        Self {
            current_mean: first_observation,
            n_observations: 1,
        }
    }

    fn update(&mut self, new_observation: &ArrayBase<impl Data<Elem = f64>, Ix1>) {
        self.n_observations += 1;
        let shift =
            (new_observation - &self.current_mean).mapv_into(|x| x / self.n_observations as f64);
        self.current_mean += &shift;
    }
}

fn update_cluster_memberships(
    centroids: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    observations: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    cluster_memberships: &mut ArrayBase<impl DataMut<Elem = usize>, Ix1>,
) {
    Zip::from(observations.axis_iter(Axis(0)))
        .and(cluster_memberships)
        .par_apply(|observation, cluster_membership| {
            *cluster_membership = closest_centroid(&centroids, &observation)
        });
}

fn compute_cluster_memberships(
    centroids: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array1<usize> {
    observations.map_axis(Axis(1), |observation| {
        closest_centroid(&centroids, &observation)
    })
}

fn closest_centroid(
    centroids: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    observation: &ArrayBase<impl Data<Elem = f64>, Ix1>,
) -> usize {
    let mut iterator = centroids.genrows().into_iter().peekable();

    let first_centroid = iterator
        .peek()
        .expect("There has to be at least one centroid");
    let (mut closest_index, mut minimum_distance) = (
        0,
        first_centroid
            .sq_l2_dist(&observation)
            .expect("Failed to compute distance"),
    );

    for (centroid_index, centroid) in iterator.enumerate() {
        let distance = centroid
            .sq_l2_dist(&observation)
            .expect("Failed to compute distance");
        if distance < minimum_distance {
            closest_index = centroid_index;
            minimum_distance = distance;
        }
    }
    closest_index
}

fn get_random_centroids<S>(
    n_clusters: usize,
    observations: &ArrayBase<S, Ix2>,
    rng: &mut impl Rng,
) -> Array2<f64>
where
    S: Data<Elem = f64>,
{
    let (n_samples, _) = observations.dim();
    let indices = rand::seq::index::sample(rng, n_samples, n_clusters).into_vec();
    observations.select(Axis(0), &indices)
}
