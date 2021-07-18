use linfa::dataset::Records;
use linfa::traits::Transformer;
use linfa_clustering::{generate_blobs, Optics};
use ndarray::{array, Array, Array2};
use ndarray_npy::write_npy;
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::iter::FromIterator;

// A routine DBScan task: build a synthetic dataset, predict clusters for it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    let expected_centroids = array![[10., 10.], [5., 5.], [20., 30.], [-20., 30.],];
    let n = 100;
    let dataset: Array2<f64> = generate_blobs(n, &expected_centroids, &mut rng);

    // Configure our training algorithm
    let min_points = 3;

    println!(
        "Performing Optics Analysis with #{} data points grouped in {} blobs",
        dataset.nsamples(),
        n
    );

    // Perform OPTICS analysis with minimum points for a cluster neighborhood set to 3
    let analysis = Optics::params(min_points)
        .tolerance(3.0)
        .transform(dataset.view())
        .unwrap();

    println!();
    println!("Result: ");
    for sample in analysis.iter() {
        println!("{:?}", sample);
    }
    println!();

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("dataset.npy", &dataset).expect("Failed to write .npy file");
    write_npy(
        "reachability.npy",
        &Array::from_iter(
            analysis
                .iter()
                .map(|&x| x.reachability_distance().unwrap_or(f64::INFINITY)),
        ),
    )
    .expect("Failed to write .npy file");
    write_npy(
        "indexes.npy",
        &Array::from_iter(analysis.iter().map(|&x| x.index() as u32)),
    )
    .expect("Failed to write .npy file");
}
