use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa_nn::{
    balltree::BallTreeBuilder, distance::CommonDistance, kdtree::KdTreeBuilder,
    linear::LinearSearchBuilder, NearestNeighbourBuilder,
};
use ndarray::{Array1, Array2};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_isaac::Isaac64Rng;

fn nn_build_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let mut benchmark = c.benchmark_group("nn_build");
    let n_features = 3;
    let algorithms: &[(Box<dyn NearestNeighbourBuilder<f64>>, _)] = &[
        (Box::new(KdTreeBuilder::new()), "kdtree"),
        (Box::new(BallTreeBuilder::new()), "balltree"),
    ];

    for &n_points in &[1000, 5000, 10000] {
        let rng = &mut rng;
        let points = Array2::random_using((n_points, n_features), Uniform::new(-500., 500.), rng);

        for (alg, name) in algorithms {
            benchmark.bench_with_input(
                BenchmarkId::new(*name, format!("{}", n_points)),
                &points,
                |bencher, points| {
                    bencher.iter(|| {
                        alg.from_batch(points, CommonDistance::SqL2Dist).unwrap();
                    });
                },
            );
        }
    }
}

fn k_nearest_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let mut benchmark = c.benchmark_group("k_nearest");
    let n_features = 3;
    let distr = Uniform::new(-500., 500.);

    let algorithms: &[(Box<dyn NearestNeighbourBuilder<f64>>, _)] = &[
        (Box::new(LinearSearchBuilder::new()), "linear search"),
        (Box::new(KdTreeBuilder::new()), "kdtree"),
        (Box::new(BallTreeBuilder::new()), "balltree"),
    ];

    for &(n_points, k) in &[(10000, 10), (50000, 100), (50000, 1000)] {
        let pt = Array1::random_using(n_features, distr, &mut rng);
        let points = Array2::random_using((n_points, n_features), distr, &mut rng);

        for (alg, name) in algorithms {
            let nn = alg.from_batch(&points, CommonDistance::SqL2Dist).unwrap();
            benchmark.bench_with_input(
                BenchmarkId::new(*name, format!("{}-{}", n_points, k)),
                &k,
                |bencher, &k| {
                    bencher.iter(|| {
                        nn.k_nearest(pt.view(), k).unwrap();
                    });
                },
            );
        }
    }
}

fn within_range_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let mut benchmark = c.benchmark_group("within_range");
    let n_features = 3;
    let distr = Uniform::new(-50., 50.);

    let algorithms: &[(Box<dyn NearestNeighbourBuilder<f64>>, _)] = &[
        (Box::new(LinearSearchBuilder::new()), "linear search"),
        (Box::new(KdTreeBuilder::new()), "kdtree"),
        (Box::new(BallTreeBuilder::new()), "balltree"),
    ];

    for &(n_points, range) in &[(50000, 50.0), (50000, 100.0)] {
        let pt = Array1::random_using(n_features, distr, &mut rng);
        let points = Array2::random_using((n_points, n_features), distr, &mut rng);

        for (alg, name) in algorithms {
            let nn = alg.from_batch(&points, CommonDistance::SqL2Dist).unwrap();
            benchmark.bench_with_input(
                BenchmarkId::new(*name, format!("{}-{}", n_points, range)),
                &range,
                |bencher, &range| {
                    bencher.iter(|| {
                        nn.within_range(pt.view(), range).unwrap();
                    });
                },
            );
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = nn_build_bench, k_nearest_bench, within_range_bench
}
criterion_main!(benches);
