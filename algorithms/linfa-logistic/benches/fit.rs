//! Fitting binary and multinomial logistic regressions with the default settings:
//!
//!     cargo bench -p linfa-logistic --bench fit
//!
//! Uses only the public API.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;
use linfa::prelude::*;
use linfa_logistic::{LogisticRegression, MultiLogisticRegression};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{rand::SeedableRng, rand_distr::Normal, rand_distr::Uniform, RandomExt};
use rand_xoshiro::Xoshiro256Plus;

const N_SAMPLES: usize = 10_000;
const N_CLASSES: usize = 10;
const N_FEATURES: [usize; 4] = [2, 8, 32, 128];

/// Points in [-1, 1]^d and the scores of `n_classes` random linear models plus noise, scaled so
/// the classes overlap: a problem that takes the optimiser a realistic number of iterations.
fn problem(
    n: usize,
    d: usize,
    n_classes: usize,
    rng: &mut Xoshiro256Plus,
) -> (Array2<f64>, Array2<f64>) {
    let x = Array2::random_using((n, d), Uniform::new(-1., 1.), rng);
    let w = Array2::random_using(
        (d, n_classes),
        Normal::new(0., 3.0 / (d as f64).sqrt()).unwrap(),
        rng,
    );
    let noise = Array2::random_using((n, n_classes), Normal::new(0., 1.).unwrap(), rng);
    let scores = x.dot(&w) + noise;
    (x, scores)
}

/// Defines `$name`, benchmarking fits in `$F`. A macro rather than a generic function, as the
/// float bound of the models isn't public.
macro_rules! bench_type {
    ($name:ident, $F:ty, $ty:literal) => {
        fn $name(c: &mut Criterion) {
            type F = $F;
            let ty = $ty;

            let mut rng = Xoshiro256Plus::seed_from_u64(42);
            let n = N_SAMPLES;

            let mut group = c.benchmark_group(format!("Logistic fit binary {ty} n={n}"));
            config::set_default_benchmark_configs(&mut group);
            group.sample_size(10);
            for d in N_FEATURES {
                let (x, scores) = problem(n, d, 1, &mut rng);
                let y: Array1<bool> = scores.column(0).mapv(|s| s > 0.0);
                let dataset = Dataset::new(x.mapv(|v| v as F), y);
                group.bench_with_input(BenchmarkId::from_parameter(d), &dataset, |b, dataset| {
                    b.iter(|| LogisticRegression::<F>::default().fit(dataset).unwrap())
                });
            }
            group.finish();

            let k = N_CLASSES;
            let mut group = c.benchmark_group(format!("Logistic fit multinomial {ty} n={n} k={k}"));
            config::set_default_benchmark_configs(&mut group);
            group.sample_size(10);
            for d in N_FEATURES {
                let (x, scores) = problem(n, d, k, &mut rng);
                let y: Array1<usize> = scores
                    .axis_iter(Axis(0))
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .fold(
                                (0, f64::MIN),
                                |best, (i, &s)| if s > best.1 { (i, s) } else { best },
                            )
                            .0
                    })
                    .collect();
                let dataset = Dataset::new(x.mapv(|v| v as F), y);
                group.bench_with_input(BenchmarkId::from_parameter(d), &dataset, |b, dataset| {
                    b.iter(|| {
                        MultiLogisticRegression::<F>::default()
                            .fit(dataset)
                            .unwrap()
                    })
                });
            }
            group.finish();
        }
    };
}

bench_type!(bench_f32, f32, "f32");
bench_type!(bench_f64, f64, "f64");

fn bench(c: &mut Criterion) {
    bench_f32(c);
    bench_f64(c);
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
