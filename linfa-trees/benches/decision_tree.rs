use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa_clustering::generate_blobs;
use linfa_trees::{DecisionTree, DecisionTreeParams, SplitQuality};
use ndarray::{Array, Array2};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;
use std::iter::FromIterator;

fn decision_tree_bench(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(42);
    let training_set_sizes = vec![10, 20, 30, 40, 50];

    let n_classes: u64 = 4;
    let n_features = 4;

    let hyperparams = DecisionTreeParams::new(n_classes as u64)
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(50))
        .min_samples_split(5)
        .min_samples_leaf(5);

    let mut group = c.benchmark_group("decision_tree");
    for n in training_set_sizes.iter() {
        let centroids = Array2::random_using(
            (n_classes as usize, n_features),
            Uniform::new(-30., 30.),
            &mut rng,
        );
        let train_x = generate_blobs(*n, &centroids, &mut rng);

        let train_y = Array::from_iter(
            (0..n_classes)
                .map(|x| std::iter::repeat(x).take(*n).collect::<Vec<u64>>())
                .flatten(),
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(train_x, train_y),
            |b, (x, y)| b.iter(|| black_box(DecisionTree::fit(hyperparams.build(), &x, &y))),
        );
    }
    group.finish();
}

criterion_group!(benches, decision_tree_bench);
criterion_main!(benches);
