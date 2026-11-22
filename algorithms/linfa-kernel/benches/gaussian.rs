//! Dense Gaussian kernel construction:
//!
//!     cargo bench -p linfa-kernel --bench gaussian

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use linfa::benchmarks::config;
use linfa::traits::Transformer;
use linfa::Float;
use linfa_kernel::{Kernel, KernelMethod, KernelType};
use ndarray::Array2;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_xoshiro::Xoshiro256Plus;

fn bench_type<F: Float>(c: &mut Criterion, ty: &str) {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    for n in [500, 2_000] {
        let mut group = c.benchmark_group(format!("Gaussian kernel {ty} n={n}"));
        config::set_default_benchmark_configs(&mut group);
        group.throughput(Throughput::Elements((n * n) as u64));

        for d in [2, 16, 64, 120] {
            let x: Array2<F> =
                Array2::random_using((n, d), Uniform::new(0., 1.), &mut rng).mapv(F::cast);
            // The mean squared distance between uniform points in [0, 1)^d is d/6: with
            // eps = d/3 the exponents average -1/2, a typical bandwidth.
            let params = Kernel::params()
                .kind(KernelType::Dense)
                .method(KernelMethod::Gaussian(F::cast(d as f64 / 3.0)));

            group.bench_with_input(BenchmarkId::from_parameter(d), &x, |b, x| {
                b.iter(|| params.transform(x.view()))
            });
        }
        group.finish();
    }
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
