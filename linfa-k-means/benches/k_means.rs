use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, ParameterizedBenchmark,
    PlotConfiguration,
};
use linfa_k_means::k_means;
use ndarray::{s, Array, Array2, ArrayView1, ArrayView2};
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
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

fn k_means_bench(c: &mut Criterion) {
    let n_clusters = 4;
    let n_features = 3;
    let tolerance = 1e-3;
    let max_n_iterations = 1000;
    let cluster_sizes = vec![10, 100, 1000, 10000];
    let mut rng = Isaac64Rng::seed_from_u64(40);

    let benchmark = ParameterizedBenchmark::new(
        "naive_k_means",
        move |bencher, &cluster_size| {
            let centroids =
                Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), &mut rng);
            let dataset = generate_dataset(cluster_size, centroids.view(), &mut rng);
            bencher.iter(|| {
                black_box(k_means(
                    n_clusters,
                    &dataset,
                    &mut rng,
                    tolerance,
                    max_n_iterations,
                ))
            });
        },
        cluster_sizes,
    )
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("naive_k_means", benchmark);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = k_means_bench
}
criterion_main!(benches);
