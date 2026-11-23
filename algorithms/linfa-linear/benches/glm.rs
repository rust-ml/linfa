//! Fitting generalized linear models (`TweedieRegressor`) with the default settings:
//!
//!     cargo bench -p linfa-linear --bench glm
//!
//! Uses only the public API.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;
use linfa::prelude::*;
use linfa_linear::{Link, TweedieRegressor};
use ndarray::{Array1, Array2};
use ndarray_rand::{rand::SeedableRng, rand_distr::Normal, RandomExt};
use rand_xoshiro::Xoshiro256Plus;

const N_SAMPLES: usize = 10_000;
const N_FEATURES: [usize; 3] = [8, 32, 128];

fn bench(c: &mut Criterion) {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    let n = N_SAMPLES;

    let mut group = c.benchmark_group(format!("GLM fit n={n}"));
    config::set_default_benchmark_configs(&mut group);
    group.sample_size(10);
    for d in N_FEATURES {
        // Positive targets, log-linear in the features plus noise: valid for every distribution
        let x = Array2::random_using((n, d), Normal::new(0., 1.).unwrap(), &mut rng);
        let w = Array1::random_using(
            d,
            Normal::new(0., 0.5 / (d as f64).sqrt()).unwrap(),
            &mut rng,
        );
        let noise = Array1::random_using(n, Normal::new(0., 0.3).unwrap(), &mut rng);
        let dataset = Dataset::new(x.clone(), (x.dot(&w) + noise).mapv(f64::exp));

        for (name, power, link) in [
            ("normal", 0., Link::Identity),
            ("poisson", 1., Link::Log),
            ("gamma", 2., Link::Log),
        ] {
            group.bench_with_input(BenchmarkId::new(name, d), &dataset, |b, dataset| {
                b.iter(|| {
                    TweedieRegressor::params()
                        .power(power)
                        .link(link)
                        .fit(dataset)
                        .unwrap()
                })
            });
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
