use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;
use linfa::traits::Transformer;
use linfa_kernel::{Kernel, KernelMethod, KernelType};
use ndarray::Array2;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_xoshiro::Xoshiro256Plus;

/// Dense kernel construction is quadratic in both time and memory, so the
/// sample sizes stay below the workspace-wide defaults: a 20_000 sample dense
/// kernel would already need ~3.2 GB just to hold the matrix.
const SIZES: [usize; 3] = [1_000, 2_000, 4_000];
const N_FEATURES: [usize; 2] = [3, 8];

fn dataset(n_samples: usize, n_features: usize) -> Array2<f64> {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    Array2::random_using((n_samples, n_features), Uniform::new(-1.0, 1.0), &mut rng)
}

fn build_dense(dataset: &Array2<f64>, method: KernelMethod<f64>) {
    Kernel::params()
        .kind(KernelType::Dense)
        .method(method)
        .transform(dataset.view());
}

fn bench(c: &mut Criterion) {
    let methods = [
        (KernelMethod::Gaussian(0.5), "Gaussian"),
        (KernelMethod::Linear, "Linear"),
        (KernelMethod::Polynomial(1.0, 3.0), "Polynomial"),
    ];

    let mut group = c.benchmark_group("Kernel");
    config::set_default_benchmark_configs(&mut group);

    for (method, name) in methods {
        for n_features in N_FEATURES {
            for n_samples in SIZES {
                let data = dataset(n_samples, n_features);

                group.bench_with_input(
                    BenchmarkId::new(
                        format!("Dense-{name}-{n_features}feats"),
                        format!("{n_samples}samples"),
                    ),
                    &data,
                    |b, data| b.iter(|| build_dense(data, method.clone())),
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
