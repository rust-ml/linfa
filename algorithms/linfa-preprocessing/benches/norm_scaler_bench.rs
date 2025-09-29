use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;
use linfa::traits::Transformer;
use linfa_datasets::generate::make_dataset;
use linfa_preprocessing::norm_scaling::NormScaler;
use statrs::distribution::{DiscreteUniform, Laplace};

fn bench(c: &mut Criterion) {
    let mut benchmark = c.benchmark_group("norm scaler");
    config::set_default_benchmark_configs(&mut benchmark);
    let size = 10000;
    let feat_distr = Laplace::new(0.5, 5.).unwrap();
    let target_distr = DiscreteUniform::new(0, 5).unwrap();

    for (scaler, fn_name) in [
        (NormScaler::l2(), "l2 scaler"),
        (NormScaler::l1(), "l1 scaler"),
        (NormScaler::max(), "max scaler"),
    ] {
        for nfeatures in (10..100).step_by(10) {
            let dataset = make_dataset(size, nfeatures, 1, feat_distr, target_distr);
            benchmark.bench_function(
                BenchmarkId::new(fn_name, format!("{nfeatures}x{size}")),
                |bencher| {
                    bencher.iter(|| {
                        scaler.transform(black_box(dataset.view()));
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
    targets = bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, bench);

criterion_main!(benches);
