use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::prelude::Predict;
use linfa::traits::FitWith;
use linfa::{Dataset, DatasetBase, ParamGuard};
use linfa_ftrl::Ftrl;
use ndarray::{Array1, Array2};
use ndarray_rand::{
    rand::distributions::Uniform, rand::rngs::SmallRng, rand::SeedableRng, RandomExt,
};
use std::time::Duration;

fn fit_without_prior_model(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let params = Ftrl::params();
    let mut group = c.benchmark_group("Ftrl with no initial model");
    let sizes: Vec<(usize, usize)> = vec![(10, 1_000), (50, 5_000), (100, 10_000)];

    for (nfeatures, nrows) in sizes.iter() {
        let dataset = get_dataset(&mut rng, *nrows, *nfeatures);
        group.bench_function(
            BenchmarkId::new("training on ", format!("dataset {}x{}", nfeatures, nrows)),
            |bencher| {
                bencher.iter(|| {
                    params.fit_with(None, black_box(&dataset)).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn fit_with_prior_model(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let params = Ftrl::params();
    let valid_params = params.clone().check().unwrap();
    let mut group = c.benchmark_group("Ftrl incremental model training");
    let sizes: Vec<(usize, usize)> = vec![(10, 1_000), (50, 5_000), (100, 10_000)];

    for (nfeatures, nrows) in sizes.iter() {
        let model = Ftrl::new(valid_params.clone(), *nfeatures);
        let dataset = get_dataset(&mut rng, *nrows, *nfeatures);
        group.bench_function(
            BenchmarkId::new("training on ", format!("dataset {}x{}", nfeatures, nrows)),
            |bencher| {
                bencher.iter(|| {
                    let _ = params
                        .fit_with(black_box(Some(model.clone())), black_box(&dataset))
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn predict(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let params = Ftrl::params();
    let valid_params = params.clone().check().unwrap();
    let mut group = c.benchmark_group("Ftrl");
    group
        .significance_level(0.02)
        .sample_size(200)
        .measurement_time(Duration::new(10, 0))
        .confidence_level(0.97)
        .warm_up_time(Duration::new(10, 0))
        .noise_threshold(0.05);

    let sizes: Vec<(usize, usize)> = vec![(10, 1_000), (50, 5_000), (100, 10_000)];
    for (nfeatures, nrows) in sizes.iter() {
        let model = Ftrl::new(valid_params.clone(), *nfeatures);
        let dataset = get_dataset(&mut rng, *nrows, *nfeatures);
        group.bench_function(
            BenchmarkId::new("predicting on ", format!("dataset {}x{}", nfeatures, nrows)),
            |bencher| {
                bencher.iter(|| {
                    model.predict(black_box(&dataset));
                });
            },
        );
    }
}

fn to_binary(value: f32) -> bool {
    value >= 0.5
}

fn get_dataset(
    rng: &mut SmallRng,
    size: usize,
    nfeatures: usize,
) -> DatasetBase<Array2<f64>, Array1<bool>> {
    let features = Array2::random_using((size, nfeatures), Uniform::from(-30. ..30.), rng);
    let target = Array1::random_using(size, Uniform::from(0. ..1.), rng).mapv(to_binary);
    Dataset::new(features, target)
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = fit_without_prior_model, fit_with_prior_model, predict
}
criterion_main!(benches);
