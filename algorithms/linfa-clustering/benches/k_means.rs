use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::{generate_blobs, KMeans, KMeansInit};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform};
use rand_isaac::Isaac64Rng;

fn k_means_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let cluster_sizes = [(100, 4), (400, 10), (3000, 10)];
    let n_features = 3;

    let mut benchmark = c.benchmark_group("naive_k_means");
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for &(cluster_size, n_clusters) in &cluster_sizes {
        let rng = &mut rng;
        let centroids =
            Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), rng);
        let dataset = DatasetBase::from(generate_blobs(cluster_size, &centroids, rng));

        benchmark.bench_function(
            BenchmarkId::new("naive_k_means", format!("{}x{}", n_clusters, cluster_size)),
            |bencher| {
                bencher.iter(|| {
                    KMeans::params_with_rng(black_box(n_clusters), black_box(rng.clone()))
                        .init_method(KMeansInit::KMeansPlusPlus)
                        .max_n_iterations(black_box(1000))
                        .tolerance(black_box(1e-3))
                        .fit(&dataset)
                        .unwrap()
                });
            },
        );
    }

    benchmark.finish();
}

fn k_means_init_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let init_methods = [KMeansInit::KMeansPlusPlus, KMeansInit::KMeansPara];
    let cluster_sizes = [(100, 10), (3000, 10), (400, 30), (500, 100)];
    let n_features = 3;

    let mut benchmark = c.benchmark_group("k_means_init");
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for init in &init_methods {
        for &(cluster_size, n_clusters) in &cluster_sizes {
            let rng = &mut rng;
            let centroids =
                Array2::random_using((n_clusters, n_features), Uniform::new(-30., 30.), rng);
            let dataset = DatasetBase::from(generate_blobs(cluster_size, &centroids, rng));
            let mut total_cost = 0.0;
            let mut runs = 0;

            benchmark.bench_function(
                BenchmarkId::new(
                    "k_means_init",
                    format!("{:?}:{}x{}", init, n_clusters, cluster_size),
                ),
                |bencher| {
                    bencher.iter(|| {
                        // Do 1 run of KMeans with 1 iterations, so it's mostly just the init
                        // algorithm
                        let m = KMeans::params_with_rng(black_box(n_clusters), rng.clone())
                            .init_method(init.clone())
                            .max_n_iterations(1)
                            .n_runs(1)
                            .tolerance(1000.0) // Guaranteed convergence
                            .fit(&dataset)
                            .unwrap();
                        total_cost += m.cost();
                        runs += 1;
                    });
                },
            );

            println!("Average cost = {}", total_cost / runs as f64);
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = k_means_bench, k_means_init_bench
}
criterion_main!(benches);
