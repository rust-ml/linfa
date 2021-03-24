use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::{generate_blobs, KMeans, KMeansInit};
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

fn k_means_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let cluster_sizes = vec![(100, 4), (400, 10), (3000, 10)];

    let mut benchmark = c.benchmark_group("naive_k_means");
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for (cluster_size, n_clusters) in cluster_sizes {
        let rng = &mut rng;
        let n_features = 3;
        let centroids =
            Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), rng);
        let dataset = DatasetBase::from(generate_blobs(cluster_size, &centroids, rng));
        benchmark.bench_function(BenchmarkId::new("naive_k_means", cluster_size), |bencher| {
            bencher.iter(|| {
                KMeans::params_with_rng(black_box(n_clusters), black_box(rng.clone()))
                    .init_method(KMeansInit::KMeansPlusPlus)
                    .max_n_iterations(black_box(1000))
                    .tolerance(black_box(1e-3))
                    .fit(&dataset)
                    .unwrap()
            });
        });
    }

    benchmark.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = k_means_bench
}
criterion_main!(benches);
