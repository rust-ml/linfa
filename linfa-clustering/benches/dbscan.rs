use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, ParameterizedBenchmark,
    PlotConfiguration,
};
use linfa_clustering::{generate_blobs, Dbscan, DbscanHyperParams};
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

fn dbscan_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let cluster_sizes = vec![10, 100, 1000, 10000];

    let benchmark = ParameterizedBenchmark::new(
        "dbscan",
        move |bencher, &cluster_size| {
            let min_points = 4;
            let n_features = 3;
            let centroids =
                Array2::random_using((min_points, n_features), Uniform::new(-30., 30.), &mut rng);
            let dataset = generate_blobs(cluster_size, &centroids, &mut rng);
            let hyperparams = DbscanHyperParams::new(min_points).tolerance(1e-3).build();
            bencher.iter(|| black_box(Dbscan::predict(&hyperparams, &dataset)));
        },
        cluster_sizes,
    )
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("dbscan", benchmark);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = dbscan_bench
}
criterion_main!(benches);
