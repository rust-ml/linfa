use linfa::prelude::*;
use linfa_clustering::generate_blobs;
use linfa_reduction::Pca;

use ndarray::array;
use ndarray_npy::write_npy;
use rand::{rngs::SmallRng, SeedableRng};

// A routine K-means task: build a synthetic dataset, fit the algorithm on it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = SmallRng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    let n = 10;
    let dataset = Dataset::from(generate_blobs(n, &expected_centroids, &mut rng));

    let embedding: Pca<f64> = Pca::params(1).fit(&dataset).unwrap();
    let embedding = embedding.predict(&dataset);

    dbg!(&embedding);

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("pca_dataset.npy", &dataset.records().view()).expect("Failed to write .npy file");
    write_npy("pca_embedding.npy", &embedding).expect("Failed to write .npy file");
}
