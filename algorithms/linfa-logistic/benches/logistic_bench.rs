use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::prelude::*;
use ndarray::{Array1, Ix1};
use rand::{Rng, SeedableRng};

const MAX_ITERATIONS: u64 = 2;

fn train_model(
    dataset: &Dataset<f32, bool, Ix1>,
) -> linfa_logistic::FittedLogisticRegression<f32, bool> {
    linfa_logistic::LogisticRegression::default()
        .max_iterations(MAX_ITERATIONS)
        .fit(dataset)
        .unwrap()
}

fn generate_categorical_data(nfeatures: usize, nsamples: usize) -> Dataset<f32, bool, Ix1> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut feature_rows: Vec<Vec<f32>> = Vec::new();
    let mut label_rows: Vec<bool> = Vec::new();
    for _ in 0..nsamples {
        let mut features = Vec::new();
        for _ in 0..nfeatures {
            let value = if rng.gen() { 1.0 } else { 0.0 };
            features.push(value);
        }
        feature_rows.push(features);
        label_rows.push(rng.gen());
    }
    linfa::Dataset::new(
        ndarray::Array2::from_shape_vec(
            (nsamples, nfeatures),
            feature_rows.into_iter().flatten().collect(),
        )
        .unwrap(),
        Array1::from_shape_vec(label_rows.len(), label_rows).unwrap(),
    )
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Logistic regression");
    group.measurement_time(std::time::Duration::from_secs(10)).sample_size(10);
    for nfeatures in [1_000] {
        for nsamples in [1_000, 10_000, 100_000, 200_000, 500_000, 1_000_000] {
            let input = generate_categorical_data(nfeatures, nsamples);
            group.bench_with_input(
                BenchmarkId::new("train_model", format!("{:e}x{:e}", nfeatures as f64, nsamples as f64)),
                &input,
                |b, dataset| {
                    b.iter(|| train_model(dataset));
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
