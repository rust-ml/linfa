use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, ParameterizedBenchmark,
    PlotConfiguration,
};
use linfa::traits::Fit;
use linfa::Dataset;
use linfa_clustering::{generate_blobs, KMeans};
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

fn k_means_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let cluster_sizes = vec![10, 100, 1000, 10000];

    let benchmark = ParameterizedBenchmark::new(
        "naive_k_means",
        move |bencher, &cluster_size| {
            let n_clusters = 4;
            let n_features = 3;
            let centroids =
                Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), &mut rng);
            let dataset = Dataset::from(generate_blobs(cluster_size, &centroids, &mut rng));
            bencher.iter(|| {
                black_box(
                    KMeans::params_with_rng(n_clusters, rng.clone())
                        .max_n_iterations(1000)
                        .tolerance(1e-3)
                        .build()
                        .fit(&dataset),
                )
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
