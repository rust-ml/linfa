//! Fitting elastic nets with the default settings, on correlated features:
//!
//!     cargo bench -p linfa-elasticnet --bench fit
//!
//! Uses only the public API.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;
use linfa::prelude::*;
use linfa_elasticnet::{ElasticNet, MultiTaskElasticNet};
use ndarray::Array2;
use ndarray_rand::{rand::SeedableRng, rand_distr::Normal, RandomExt};
use rand_xoshiro::Xoshiro256Plus;

const N_SAMPLES: usize = 5_000;
const N_TASKS: usize = 4;
const N_FEATURES: [usize; 3] = [8, 32, 128];
/// Correlation between every pair of features. Correlated features are the usual case and take
/// coordinate descent many passes; independent ones converge in a few.
const RHO: f64 = 0.8;

/// Features with correlation `RHO` (a shared factor plus noise) and `n_targets` targets, linear
/// in the features plus noise.
fn problem(
    n: usize,
    d: usize,
    n_targets: usize,
    rng: &mut Xoshiro256Plus,
) -> (Array2<f64>, Array2<f64>) {
    let normal = Normal::new(0., 1.).unwrap();
    let factor = Array2::random_using((n, 1), normal, rng);
    let noise = Array2::random_using((n, d), normal, rng);
    let x = &factor * RHO.sqrt() + &noise * (1.0 - RHO).sqrt();
    let w = Array2::random_using((d, n_targets), normal, rng);
    let y = x.dot(&w) + Array2::random_using((n, n_targets), Normal::new(0., 0.5).unwrap(), rng);
    (x, y)
}

fn bench_type<F: linfa::Float>(c: &mut Criterion, ty: &str) {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    let n = N_SAMPLES;
    let penalty = F::cast(0.1);

    let mut group = c.benchmark_group(format!("ElasticNet fit {ty} n={n}"));
    config::set_default_benchmark_configs(&mut group);
    group.sample_size(10);
    for d in N_FEATURES {
        let (x, y) = problem(n, d, 1, &mut rng);
        let dataset = Dataset::new(x.mapv(F::cast), y.column(0).mapv(F::cast));
        group.bench_with_input(BenchmarkId::from_parameter(d), &dataset, |b, dataset| {
            b.iter(|| ElasticNet::params().penalty(penalty).fit(dataset).unwrap())
        });
    }
    group.finish();

    let k = N_TASKS;
    let mut group = c.benchmark_group(format!("MultiTaskElasticNet fit {ty} n={n} k={k}"));
    config::set_default_benchmark_configs(&mut group);
    group.sample_size(10);
    for d in N_FEATURES {
        let (x, y) = problem(n, d, k, &mut rng);
        let dataset = Dataset::new(x.mapv(F::cast), y.mapv(F::cast));
        group.bench_with_input(BenchmarkId::from_parameter(d), &dataset, |b, dataset| {
            b.iter(|| {
                MultiTaskElasticNet::params()
                    .penalty(penalty)
                    .fit(dataset)
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench(c: &mut Criterion) {
    bench_type::<f32>(c, "f32");
    bench_type::<f64>(c, "f64");
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
