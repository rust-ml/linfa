use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, ParameterizedBenchmark,
    PlotConfiguration,
};
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::{generate_blobs, GaussianMixtureModel};
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

fn gaussian_mixture_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let cluster_sizes = vec![10, 100, 1000, 10000];

    let benchmark = ParameterizedBenchmark::new(
        "gaussian_mixture",
        move |bencher, &cluster_size| {
            let n_clusters = 4;
            let n_features = 3;
            let centroids =
                Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), &mut rng);
            let dataset : DatasetBase<_,_> = (generate_blobs(cluster_size, &centroids, &mut rng),()).into();
            bencher.iter(|| {
                black_box(
                    GaussianMixtureModel::params(n_clusters)
                        .with_rng(rng.clone())
                        .with_tolerance(1e-3)
                        .with_max_n_iterations(1000)
                        .fit(&dataset)
                        .expect("GMM fitting fail"),
                )
            });
        },
        cluster_sizes,
    )
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("gaussian_mixture", benchmark);
}

criterion_group! {
  name = benches;
  config = Criterion::default();
  targets = gaussian_mixture_bench
}
criterion_main!(benches);
