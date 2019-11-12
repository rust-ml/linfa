use linfa_k_means::KMeans;
use ndarray::{array, s, Array, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_npy::write_npy;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

pub fn generate_dataset(
    cluster_size: usize,
    centroids: ArrayView2<f64>,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let (n_centroids, n_features) = centroids.dim();
    let mut dataset: Array2<f64> = Array2::zeros((n_centroids * cluster_size, n_features));

    for (cluster_index, centroid) in centroids.genrows().into_iter().enumerate() {
        let cluster = generate_cluster(cluster_size, centroid, rng);

        let indexes = s![
            cluster_index * cluster_size..(cluster_index + 1) * cluster_size,
            ..
        ];
        dataset.slice_mut(indexes).assign(&cluster);
    }
    dataset
}

pub fn generate_cluster(
    n_observations: usize,
    centroid: ArrayView1<f64>,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let shape = (n_observations, centroid.len());
    let origin_cluster: Array2<f64> = Array::random_using(shape, StandardNormal, rng);
    origin_cluster + centroid
}

fn main() {
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    let n = 10000;

    let mut rng = Isaac64Rng::seed_from_u64(42);
    let max_n_iterations = 200;
    let tolerance = 1e-5;
    let n_clusters = expected_centroids.len_of(Axis(0));

    let dataset = generate_dataset(n, expected_centroids.view(), &mut rng);

    let mut model = KMeans::new(Some(tolerance), Some(max_n_iterations), &mut rng);
    model.fit(n_clusters, &dataset);
    let cluster_memberships = model.predict(&dataset);

    write_npy("clustered_dataset.npy", dataset).expect("Failed to write .npy file");
    write_npy(
        "clustered_memberships.npy",
        cluster_memberships.map(|&x| x as u64),
    )
    .expect("Failed to write .npy file");
}
