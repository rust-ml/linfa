//! Compares fitting binary logistic regression on a sparse (CSR) feature matrix against fitting
//! it on the same matrix stored densely. Both fits solve the same problem with the same number of
//! solver iterations, so the difference is the cost of the matrix products.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Ix1};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use sprs::{CsMat, TriMat};

/// Random binary design matrix with roughly `density * n_features` stored ones per row, and
/// labels from a logistic model with random weights. Seeded for reproducibility.
fn make_problem(n_samples: usize, n_features: usize, density: f64) -> (CsMat<f64>, Array1<bool>) {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    let weights: Vec<f64> = (0..n_features).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let nnz_per_row = ((density * n_features as f64).round() as usize).max(1);
    let mut triplets = TriMat::new((n_samples, n_features));
    let mut labels = Array1::from_elem(n_samples, false);
    for (row, label) in labels.iter_mut().enumerate() {
        let mut z = 0.0;
        for _ in 0..nnz_per_row {
            let col = rng.gen_range(0..n_features);
            triplets.add_triplet(row, col, 1.0);
            z += weights[col];
        }
        *label = rng.gen::<f64>() < 1.0 / (1.0 + (-z).exp());
    }
    (triplets.to_csr(), labels)
}

fn fit_dense(dataset: &Dataset<f64, bool, Ix1>) {
    let _ = LogisticRegression::default()
        .max_iterations(50)
        .fit(dataset)
        .unwrap();
}

fn fit_sparse(dataset: &DatasetBase<CsMat<f64>, Array1<bool>>) {
    let _ = LogisticRegression::default()
        .max_iterations(50)
        .fit(dataset)
        .unwrap();
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linfa_logistic_sparse_vs_dense");
    config::set_default_benchmark_configs(&mut group);
    // A dense fit at the largest size takes seconds, so the default of 200 samples per
    // benchmark would take hours; 10 is criterion's minimum.
    group.sample_size(10);

    for n_features in [1_000, 5_000] {
        for density in [0.01, 0.1] {
            for n_samples in [1_000, 10_000, 20_000] {
                let (x, y) = make_problem(n_samples, n_features, density);
                let suffix = format!("{n_features}feats-{}pctDense", density * 100.0);

                let dense: Dataset<f64, bool, Ix1> = Dataset::new(x.to_dense(), y.clone());
                group.bench_with_input(
                    BenchmarkId::new(format!("Dense-{suffix}"), n_samples),
                    &dense,
                    |b, dataset| b.iter(|| fit_dense(dataset)),
                );
                drop(dense);

                let sparse = DatasetBase::new(x, y);
                group.bench_with_input(
                    BenchmarkId::new(format!("Csr-{suffix}"), n_samples),
                    &sparse,
                    |b, dataset| b.iter(|| fit_sparse(dataset)),
                );
            }
        }
    }
    group.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, bench);

criterion_main!(benches);
