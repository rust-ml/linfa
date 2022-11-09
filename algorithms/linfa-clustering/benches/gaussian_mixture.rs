use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::GaussianMixtureModel;
use linfa_datasets::generate;
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
#[cfg(not(target_os = "windows"))]
use pprof::criterion::{Output, PProfProfiler};
use rand_xoshiro::Xoshiro256Plus;

fn gaussian_mixture_bench(c: &mut Criterion) {
    let mut rng = Xoshiro256Plus::seed_from_u64(40);
    let cluster_sizes = vec![10, 100, 1000, 10000];

    let mut benchmark = c.benchmark_group("gaussian_mixture");
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

criterion_group! {
  name = benches;
  config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
  targets = gaussian_mixture_bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, gaussian_mixture_bench);
criterion_main!(benches);
