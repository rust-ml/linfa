use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use linfa::benchmarks::config;
use linfa::prelude::{ParamGuard, Transformer};
use linfa_clustering::Dbscan;
use linfa_datasets::generate;
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_xoshiro::Xoshiro256Plus;

fn dbscan_bench(c: &mut Criterion) {
    let mut rng = Xoshiro256Plus::seed_from_u64(40);
    let cluster_sizes = vec![10, 100, 1000, 10000];

    let (sample_size, measurement_time, confidence_level, warm_up_time, noise_threshold) =
        config::get_default_benchmark_configs();

    let mut benchmark = c.benchmark_group("dbscan");
    benchmark
        .sample_size(sample_size)
        .measurement_time(measurement_time)
        .confidence_level(confidence_level)
        .warm_up_time(warm_up_time)
        .noise_threshold(noise_threshold)
        .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for cluster_size in cluster_sizes {
        let rng = &mut rng;
        benchmark.bench_with_input(
            BenchmarkId::new("dbscan", cluster_size),
            &cluster_size,
            move |bencher, &cluster_size| {
                let min_points = 4;
                let n_features = 3;
                let tolerance = 0.3;
                let centroids =
                    Array2::random_using((min_points, n_features), Uniform::new(-30., 30.), rng);
                let dataset = generate::blobs(cluster_size, &centroids, rng);

                bencher.iter(|| {
                    black_box(
                        Dbscan::params(min_points)
                            .tolerance(tolerance)
                            .check_unwrap()
                            .transform(&dataset),
                    )
                });
            },
        );
    }
    benchmark.finish()
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = dbscan_bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, dbscan_bench);

criterion_main!(benches);
