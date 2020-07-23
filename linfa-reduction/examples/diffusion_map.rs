use linfa_reduction::utils::generate_convoluted_rings;
use linfa_reduction::kernel::{IntoKernel, SparsePolynomialKernel};
use ndarray_npy::write_npy;
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;

// A routine K-means task: build a synthetic dataset, fit the algorithm on it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let n = 3000;

    // generate three convoluted rings
    let dataset = generate_convoluted_rings(&[(0.0, 3.0), (10.0, 13.0), (20.0, 23.0)], n, &mut rng);

    // generate sparse polynomial kernel with k = 14, c = 5 and d = 2
    let diffusion_map = SparsePolynomialKernel::new(&dataset, 14, 5.0, 2.0)
        .reduce_fixed(4);

    // get embedding
    let embedding = diffusion_map.embedding();

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("diffusion_map_dataset.npy", dataset).expect("Failed to write .npy file");
    write_npy(
        "diffusion_map_embedding.npy",
        embedding
    )
    .expect("Failed to write .npy file");
}
