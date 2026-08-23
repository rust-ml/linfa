use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;
use linfa::traits::Fit;
use linfa::traits::Transformer;
use linfa_datasets::generate::make_dataset;
use linfa_preprocessing::whitening::Whitener;
use statrs::distribution::{DiscreteUniform, Laplace};
use std::time::Duration;

fn bench(c: &mut Criterion) {
    let mut benchmark = c.benchmark_group("whitening");
    config::set_default_benchmark_configs(&mut benchmark);
    let size = 10000;
    let feat_distr = Laplace::new(0.5, 5.).unwrap();
    let target_distr = DiscreteUniform::new(0, 5).unwrap();

    for (whitener, fn_name) in [
        (Whitener::cholesky(), "cholesky"),
        (Whitener::zca(), "zca"),
        (Whitener::pca(), "pca"),
    ] {
        for nfeatures in (10..100).step_by(10) {
            let dataset = make_dataset(size, nfeatures, 1, feat_distr, target_distr);
            benchmark.bench_function(
                BenchmarkId::new(fn_name, format!("{nfeatures}x{size}")),
                |bencher| {
                    bencher.iter(|| {
                        whitener
                            .fit(black_box(&dataset))
                            .unwrap()
                            .transform(black_box(dataset.view()));
                    });
                },
            );
        }
    }
}

/// Sweeps the sample count with the feature count held fixed, timing `fit` on its own.
///
/// The group above varies the feature count at a fixed sample count and times `fit`
/// together with `transform`, which is linear in the sample count whatever `fit` does.
/// Isolating `fit` over a doubling sweep is what shows how each method scales with the
/// number of samples.
fn bench_fit_by_nsamples(c: &mut Criterion) {
    let mut benchmark = c.benchmark_group("whitening_fit");
    config::set_default_benchmark_configs(&mut benchmark);
    // A single `fit` here runs into the seconds, so the shared 200-sample default would
    // put this group in the tens of minutes.
    benchmark
        .sample_size(20)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(5));

    let nfeatures = 64;
    let feat_distr = Laplace::new(0.5, 5.).unwrap();
    let target_distr = DiscreteUniform::new(0, 5).unwrap();

    for nsamples in [2500, 5000, 10000, 20000, 40000] {
        let dataset = make_dataset(nsamples, nfeatures, 1, feat_distr, target_distr);
        for (whitener, fn_name) in [
            (Whitener::cholesky(), "cholesky"),
            (Whitener::zca(), "zca"),
            (Whitener::pca(), "pca"),
        ] {
            benchmark.bench_function(
                BenchmarkId::new(fn_name, format!("{nfeatures}x{nsamples}")),
                |bencher| {
                    bencher.iter(|| {
                        whitener.fit(black_box(&dataset)).unwrap();
                    });
                },
            );
        }
    }
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = bench, bench_fit_by_nsamples
}
#[cfg(target_os = "windows")]
criterion_group!(benches, bench, bench_fit_by_nsamples);

criterion_main!(benches);
