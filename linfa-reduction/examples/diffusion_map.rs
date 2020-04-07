use linfa_reduction::{DiffusionMap, DiffusionMapHyperParams};
use linfa_reduction::utils::generate_swissroll;
use linfa_reduction::kernel::{IntoKernel, DotKernel, SparseGaussianKernel, GaussianKernel};
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
    let n = 2000;
    let dataset = generate_swissroll(40.0, 1.0, n, &mut rng);
    //let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    //let dataset = generate_blobs(n, &expected_centroids, &mut rng);
    //let similarity = to_gaussian_similarity(&dataset, 40.0);

    //let diffusion_map = GaussianKernel::new(&dataset, 100.0)
    //    .reduce_fixed(2);

    let diffusion_map = SparseGaussianKernel::new(&dataset.clone(), 20, 100.0)
        .reduce_fixed(2);

    let embedding = diffusion_map.embedding();
    //dbg!(&embedding);

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("diffusion_map_dataset.npy", dataset).expect("Failed to write .npy file");
    write_npy(
        "diffusion_map_embedding.npy",
        embedding
    )
    .expect("Failed to write .npy file");
}
