use linfa_clustering::{generate_blobs, Dbscan, DbscanHyperParams};
use ndarray::array;
use ndarray_npy::write_npy;
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;

// A routine DBScan task: build a synthetic dataset, predict clusters for it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    let n = 10000;
    let dataset = generate_blobs(n, &expected_centroids, &mut rng);

    // Configure our training algorithm
    let min_points = 3;
    let hyperparams = DbscanHyperParams::new(min_points).tolerance(1e-5).build();

    // Infer an optimal set of centroids based on the training data distribution
    let cluster_memberships = Dbscan::predict(&hyperparams, &dataset);

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("clustered_dataset.npy", dataset).expect("Failed to write .npy file");
    write_npy(
        "clustered_memberships.npy",
        cluster_memberships.map(|&x| x.map(|c| c as i64).unwrap_or(-1)),
    )
    .expect("Failed to write .npy file");
}
