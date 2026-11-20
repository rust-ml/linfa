use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use linfa::benchmarks::config;
use linfa::prelude::Fit;
use linfa::DatasetBase;
use linfa_clustering::{KModes, KModesInit};
use ndarray::Array2;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn generate_synthetic_categorical(
    n_samples: usize,
    n_features: usize,
    n_categories: usize,
    rng: &mut Xoshiro256Plus,
) -> Array2<usize> {
    Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(0..n_categories))
}

fn k_modes_bench(c: &mut Criterion) {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    let sample_sizes = [1_000, 10_000, 20_000];
    let feature_dims = [5, 10];
    let n_clusters = 4;
    let n_categories = 5;

    let mut benchmark = c.benchmark_group("k_modes");
    config::set_default_benchmark_configs(&mut benchmark);
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n_features in &feature_dims {
        for &n_samples in &sample_sizes {
            let raw_data =
                generate_synthetic_categorical(n_samples, n_features, n_categories, &mut rng);
            let dataset = DatasetBase::from(raw_data);

            // Benchmark Cao initialization
            benchmark.bench_function(
                BenchmarkId::new("Cao", format!("{n_samples}x{n_features}")),
                |bencher| {
                    bencher.iter(|| {
                        black_box(
                            KModes::params(black_box(n_clusters))
                                .max_n_iterations(black_box(20))
                                .init_method(KModesInit::Cao)
                                .fit(&dataset)
                                .unwrap(),
                        )
                    });
                },
            );

            // Benchmark Huang initialization
            benchmark.bench_function(
                BenchmarkId::new("Huang", format!("{n_samples}x{n_features}")),
                |bencher| {
                    bencher.iter(|| {
                        black_box(
                            KModes::params(black_box(n_clusters))
                                .max_n_iterations(black_box(20))
                                .init_method(KModesInit::Huang)
                                .n_runs(1)
                                .fit(&dataset)
                                .unwrap(),
                        )
                    });
                },
            );

            // Benchmark Random initialization
            benchmark.bench_function(
                BenchmarkId::new("Random", format!("{n_samples}x{n_features}")),
                |bencher| {
                    bencher.iter(|| {
                        black_box(
                            KModes::params(black_box(n_clusters))
                                .max_n_iterations(black_box(20))
                                .init_method(KModesInit::Random)
                                .n_runs(1)
                                .fit(&dataset)
                                .unwrap(),
                        )
                    });
                },
            );
        }
    }

    benchmark.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = k_modes_bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, k_modes_bench);

criterion_main!(benches);
