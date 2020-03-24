use linfa_clustering::generate_blobs;
use linfa_reduction::{to_gaussian_similarity, DiffusionMap, DiffusionMapHyperParams};
use ndarray::array;
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
    let n = 20;
    let dataset = generate_blobs(n, &expected_centroids, &mut rng);
    let similarity = to_gaussian_similarity(&dataset, 40.0);

    let kernel = GaussianKernel::new(dataset, 40);

    // Configure our training algorithm
    let hyperparams = DiffusionMapHyperParams::new(4)
        .steps(1)
        .build();

    // Infer an optimal set of centroids based on the training data distribution and assign optimal
    // indices to clusters
    let diffusion_map = DiffusionMap::project(hyperparams, similarity);
    dbg!(&diffusion_map.estimate_clusters());

    let embedding = diffusion_map.embedding();
    dbg!(&embedding);

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("diffusion_map_dataset.npy", dataset).expect("Failed to write .npy file");
    write_npy(
        "diffusion_map_embedding.npy",
        embedding
    )
    .expect("Failed to write .npy file");
}
