use linfa_clustering::{generate_blobs, to_gaussian_similarity, SpectralClustering, SpectralClusteringHyperParams};
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
    let n = 5;
    let dataset = generate_blobs(n, &expected_centroids, &mut rng);
    
    let embedding = dataset
        .to_similarity(Similarity::Gaussian(50.0))
        .reduce_dimensionality(Method::DiffusionMap)
        .unwrap();

    let similarity = to_gaussian_similarity(&dataset, 50.0);

    // Configure our training algorithm
    let n_clusters = expected_centroids.len_of(Axis(0));
    let hyperparams = SpectralClusteringHyperParams::new(n_clusters, 4)
        .steps(1)
        .build();

    // Infer an optimal set of centroids based on the training data distribution and assign optimal
    // indices to clusters
    let cluster_memberships = SpectralClustering::fit_predict(hyperparams, similarity, &mut rng);
    let cluster_memberships = cluster_memberships.indices();

    dbg!(&cluster_memberships);

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("spectral_dataset.npy", dataset).expect("Failed to write .npy file");
    write_npy(
        "spectral_memberships.npy",
        cluster_memberships.map(|&x| x as u64),
    )
    .expect("Failed to write .npy file");
}
