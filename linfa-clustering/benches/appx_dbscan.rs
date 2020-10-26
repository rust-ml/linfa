use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, ParameterizedBenchmark,
    PlotConfiguration,
};
use linfa::traits::Transformer;
use linfa_clustering::{generate_blobs, AppxDbscan};
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

fn appx_dbscan_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let cluster_sizes_and_slacks = vec![
        (10, 0.00001),
        (100, 0.00001),
        (1000, 0.00001),
        /*(10000, 0.1),*/
    ];

    let benchmark = ParameterizedBenchmark::new(
        "appx_dbscan",
        move |bencher, &cluster_size_and_slack| {
            let min_points = 4;
            let n_features = 3;
            let tolerance = 0.3;
            let centroids =
                Array2::random_using((min_points, n_features), Uniform::new(-30., 30.), &mut rng);
            let dataset = generate_blobs(cluster_size_and_slack.0, &centroids, &mut rng);
            bencher.iter(|| {
                black_box(
                    AppxDbscan::params(min_points)
                        .tolerance(tolerance)
                        .slack(cluster_size_and_slack.1)
                        .build()
                        .transform(&dataset),
                )
            });
        },
        cluster_sizes_and_slacks,
    )
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("appx_dbscan", benchmark);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = appx_dbscan_bench
}
criterion_main!(benches);
