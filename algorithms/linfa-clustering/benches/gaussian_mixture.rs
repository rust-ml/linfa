use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use linfa::benchmarks::config;
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::{GaussianMixtureModel, GmmInitMethod};
use linfa_datasets::generate;
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_xoshiro::Xoshiro256Plus;

fn gaussian_mixture_bench(c: &mut Criterion) {
    let mut rng = Xoshiro256Plus::seed_from_u64(40);
    let cluster_sizes = vec![10, 100, 1000, 10000];

    let mut benchmark = c.benchmark_group("gaussian_mixture");
    config::set_default_benchmark_configs(&mut benchmark);
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for cluster_size in cluster_sizes {
        let rng = &mut rng;
        benchmark.bench_with_input(
            BenchmarkId::new("gaussian_mixture", cluster_size),
            &cluster_size,
            move |bencher, &cluster_size| {
                let n_clusters = 4;
                let n_features = 3;
                let centroids =
                    Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), rng);
                let dataset: DatasetBase<_, _> =
                    (generate::blobs(cluster_size, &centroids, rng)).into();
                bencher.iter(|| {
                    black_box(
                        GaussianMixtureModel::params(n_clusters)
                            .with_rng(rng.clone())
                            .tolerance(1e-3)
                            .max_n_iterations(1000)
                            .fit(&dataset)
                            .expect("GMM fitting fail"),
                    )
                });
            },
        );
    }
    benchmark.finish();
}

/// The EM iterations themselves: overlapping clusters, random initialisation, and a tolerance that
/// is never met, so that every fit runs exactly `N_ITERATIONS` iterations (and reports that it
/// did not converge, which is ignored).
fn gaussian_mixture_em_bench(c: &mut Criterion) {
    const N_CLUSTERS: usize = 8;
    const BLOB_SIZE: usize = 1_250;
    const N_ITERATIONS: u64 = 20;
    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    let mut benchmark = c.benchmark_group(format!(
        "gaussian_mixture_em n={} k={N_CLUSTERS} iterations={N_ITERATIONS}",
        N_CLUSTERS * BLOB_SIZE
    ));
    config::set_default_benchmark_configs(&mut benchmark);
    benchmark.sample_size(10);

    for n_features in [2, 8, 32] {
        let centroids =
            Array2::random_using((N_CLUSTERS, n_features), Uniform::new(-2., 2.), &mut rng);
        let dataset: DatasetBase<_, _> = generate::blobs(BLOB_SIZE, &centroids, &mut rng).into();
        let init_rng = rng.clone();
        benchmark.bench_with_input(
            BenchmarkId::from_parameter(n_features),
            &dataset,
            |bencher, dataset| {
                bencher.iter(|| {
                    black_box(
                        GaussianMixtureModel::params(N_CLUSTERS)
                            .with_rng(init_rng.clone())
                            .init_method(GmmInitMethod::Random)
                            .tolerance(1e-30)
                            .max_n_iterations(N_ITERATIONS)
                            .fit(dataset)
                            .ok(),
                    )
                });
            },
        );
    }
    benchmark.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
  name = benches;
  config = config::get_default_profiling_configs();
  targets = gaussian_mixture_bench, gaussian_mixture_em_bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, gaussian_mixture_bench, gaussian_mixture_em_bench);

criterion_main!(benches);
