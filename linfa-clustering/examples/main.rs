use linfa_clustering::{generate_blobs, KMeans, KMeansHyperParams};
use ndarray::{array, Axis};
use ndarray_npy::write_npy;
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;

// A routine K-means task: build a synthetic dataset, fit the algorithm on it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    let n = 10000;
    let dataset = generate_blobs(n, &expected_centroids, &mut rng);

    // Configure our training algorithm
    let n_clusters = expected_centroids.len_of(Axis(0));
    let hyperparams = KMeansHyperParams::new(n_clusters)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .build();

    // Infer an optimal set of centroids based on the training data distribution
    let model = KMeans::fit(hyperparams, &dataset, &mut rng);

    // Assign each point to a cluster using the set of centroids found using `fit`
    let cluster_memberships = model.predict(&dataset);

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("clustered_dataset.npy", dataset).expect("Failed to write .npy file");
    write_npy(
        "clustered_memberships.npy",
        cluster_memberships.map(|&x| x as u64),
    )
    .expect("Failed to write .npy file");
}
