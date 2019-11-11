use linfa_k_means::{compute_cluster_memberships, k_means};
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
    // Let's allocate an array of the right shape to store the final dataset.
    // We will then progressively replace these zeros with the observations in each generated
    // cluster.
    let (n_centroids, n_features) = centroids.dim();
    let mut dataset: Array2<f64> = Array2::zeros((n_centroids * cluster_size, n_features));

    // There are many ways to iterate over an n-dimensional array.
    // `genrows` returns "generalised rows" or "lanes":
    // - regular rows of length `b`, if `self` is a 2-d array of shape `a` x `b`;
    // - `a` × `b` × ... × `l` rows each of length `m` for an n-dimensional array of shape
    //   `a` × `b` × ... × `l` × `m`.
    //
    // `enumerate` is an iterator method to get the element index in the iterator
    // alongside the element itself.
    for (cluster_index, centroid) in centroids.genrows().into_iter().enumerate() {
        let cluster = generate_cluster(cluster_size, centroid, rng);

        // Each cluster will contain `cluster_size` observations:
        // let's craft an index range in such a way that, at the end,
        // all zeros in `dataset` have been replaced with the observations in our
        // generated clusters.
        // You can create n-dimensional index ranges using the `s!` macro: check
        // the documentation for more details on its syntax and examples of this macro
        // in action - https://docs.rs/ndarray/0.13.0/ndarray/macro.s.html
        let indexes = s![
            cluster_index * cluster_size..(cluster_index + 1) * cluster_size,
            ..
        ];
        // `slice_mut` returns a **mutable view**: same principle of `ArrayView`, with the
        // privilege of mutable access.
        // As you might guess, you can only have one mutable view of an array going around
        // at any point in time.
        // The output type of `slice_mut` is `ArrayViewMut`, equivalent to `&mut [A]`
        // when comparing `Array` to `Vec`.
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

    let centroids = k_means(n_clusters, &dataset, &mut rng, tolerance, max_n_iterations);
    let cluster_memberships = compute_cluster_memberships(&centroids, &dataset);

    write_npy("clustered_dataset.npy", dataset).expect("Failed to write .npy file");
    write_npy(
        "clustered_memberships.npy",
        cluster_memberships.map(|&x| x as u64),
    )
    .expect("Failed to write .npy file");
}
