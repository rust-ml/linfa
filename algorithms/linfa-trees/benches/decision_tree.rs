use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{concatenate, Array, Array1, Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use rand::rngs::SmallRng;

fn generate_blobs(means: &Array2<f64>, samples: usize, mut rng: &mut SmallRng) -> Array2<f64> {
    let out = means
        .axis_iter(Axis(0))
        .map(|mean| Array::random_using((samples, 4), StandardNormal, &mut rng) + mean)
        .collect::<Vec<_>>();
    let out2 = out.iter().map(|x| x.view()).collect::<Vec<_>>();

    concatenate(Axis(0), &out2).unwrap()
}

fn decision_tree_bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    // Controls how many samples for each class are generated
    let training_set_sizes = &[100, 1000, 10000, 100000];

    let n_classes = 4;
    let n_features = 4;

    // Use the default configuration
    let hyperparams = DecisionTree::params();

    // Benchmark training time 10 times for each training sample size
    let mut group = c.benchmark_group("decision_tree");
    config::set_default_benchmark_configs(&mut group);

    for n in training_set_sizes.iter() {
        let centroids =
            Array2::random_using((n_classes, n_features), Uniform::new(-30., 30.), &mut rng);

        let train_x = generate_blobs(&centroids, *n, &mut rng);
        let train_y: Array1<usize> = (0..n_classes)
            .flat_map(|x| std::iter::repeat(x).take(*n).collect::<Vec<usize>>())
            .collect::<Array1<usize>>();
        let dataset = DatasetBase::new(train_x, train_y);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, d| {
            b.iter(|| hyperparams.fit(d))
        });
    }

    group.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = decision_tree_bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, decision_tree_bench);

criterion_main!(benches);
