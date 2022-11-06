use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::traits::Fit;
use linfa::Dataset;
use linfa_datasets::generate::make_dataset;
use linfa_pls::Algorithm;
use linfa_pls::{PlsCanonical, PlsCca, PlsRegression};
use statrs::distribution::{DiscreteUniform, Laplace};

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
    for (alg, name) in [(Algorithm::Nipals, "Nipals"), (Algorithm::Svd, "Svd")] {
        let sizes: [usize; 3] = [1_000, 10_000, 100_000];

        let feat_distr = Laplace::new(0.5, 5.).unwrap();
        let target_distr = DiscreteUniform::new(0, 5).unwrap();

        let mut pls_regression_id = "Regression-".to_string();
        pls_regression_id.push_str(name);
        let mut pls_canonical_id = "Canonical-".to_string();
        pls_canonical_id.push_str(name);
        let mut pls_cca_id = "Cca-".to_string();
        pls_cca_id.push_str(name);

        for size in sizes {
            let dataset = make_dataset(size, 5, feat_distr, target_distr);
            let targets = dataset.targets.into_shape((size, 1)).unwrap();
            let features = dataset.records;
            let dataset = Dataset::new(features, targets);
            let input = (dataset, alg);

            group.bench_with_input(
                BenchmarkId::new(&pls_regression_id, size),
                &input,
                |b, (dataset, alg)| {
                    b.iter(|| pls_regression(dataset, *alg));
                },
            );

            group.bench_with_input(
                BenchmarkId::new(&pls_canonical_id, size),
                &input,
                |b, (dataset, alg)| {
                    b.iter(|| pls_canonical(dataset, *alg));
                },
            );

            group.bench_with_input(
                BenchmarkId::new(&pls_cca_id, size),
                &input,
                |b, (dataset, alg)| {
                    b.iter(|| pls_cca(dataset, *alg));
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
