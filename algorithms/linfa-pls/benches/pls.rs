use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::traits::Fit;
use linfa::Dataset;
use linfa_datasets::generate::make_dataset;
use linfa_pls::Algorithm;
use linfa_pls::{PlsCanonical, PlsCca, PlsRegression};
#[cfg(not(target_os = "windows"))]
use pprof::criterion::{Output, PProfProfiler};
use statrs::distribution::{DiscreteUniform, Laplace};
use std::time::Duration;

#[allow(unused_must_use)]
fn pls_regression(dataset: &Dataset<f64, f64>, alg: Algorithm) {
    let model = PlsRegression::params(3)
        .scale(true)
        .max_iterations(200)
        .algorithm(alg);
    model.fit(&dataset);
}

#[allow(unused_must_use)]
fn pls_canonical(dataset: &Dataset<f64, f64>, alg: Algorithm) {
    let model = PlsCanonical::params(3)
        .scale(true)
        .max_iterations(200)
        .algorithm(alg);
    model.fit(&dataset);
}
#[allow(unused_must_use)]
fn pls_cca(dataset: &Dataset<f64, f64>, alg: Algorithm) {
    let model = PlsCca::params(3)
        .scale(true)
        .max_iterations(200)
        .algorithm(alg);
    model.fit(&dataset);
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linfa_pls");
    group
        .sample_size(200)
        .measurement_time(Duration::new(10, 0))
        .confidence_level(0.97)
        .warm_up_time(Duration::new(10, 0))
        .noise_threshold(0.05);

    let params: [(usize, usize); 4] = [(10_000, 5), (100_000, 5), (100_000, 10)];

    for (alg, name) in [(Algorithm::Nipals, "Nipals-"), (Algorithm::Svd, "Svd-")] {
        let feat_distr = Laplace::new(0.5, 5.).unwrap();
        let target_distr = DiscreteUniform::new(0, 5).unwrap();

        let mut pls_regression_id = "Regression-".to_string();
        pls_regression_id.push_str(name);
        let mut pls_canonical_id = "Canonical-".to_string();
        pls_canonical_id.push_str(name);
        let mut pls_cca_id = "Cca-".to_string();
        pls_cca_id.push_str(name);

        for (size, num_feat) in params {
            let suffix = format!("{}Feats", num_feat);
            let mut func_name = pls_regression_id.clone();
            func_name.push_str(&suffix);
            let dataset = make_dataset(size, num_feat, 1, feat_distr, target_distr);
            let input = (dataset, alg);

            group.bench_with_input(
                BenchmarkId::new(&func_name, size),
                &input,
                |b, (dataset, alg)| {
                    b.iter(|| pls_regression(dataset, *alg));
                },
            );

            let mut func_name = pls_canonical_id.clone();
            func_name.push_str(&suffix);
            group.bench_with_input(
                BenchmarkId::new(&func_name, size),
                &input,
                |b, (dataset, alg)| {
                    b.iter(|| pls_canonical(dataset, *alg));
                },
            );

            let mut func_name = pls_cca_id.clone();
            func_name.push_str(&suffix);
            group.bench_with_input(
                BenchmarkId::new(&func_name, size),
                &input,
                |b, (dataset, alg)| {
                    b.iter(|| pls_cca(dataset, *alg));
                },
            );
        }
    }
    group.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, bench);

criterion_main!(benches);
