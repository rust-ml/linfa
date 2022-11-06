use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::traits::Fit;
use linfa::Dataset;
use linfa_datasets::generate::make_dataset;
use linfa_pls::Algorithm;
use linfa_pls::PlsRegression;
use statrs::distribution::{DiscreteUniform, Laplace};

#[allow(unused_must_use)]
fn perform_pls(dataset: &Dataset<f64, f64>, alg: Algorithm) {
    let model = PlsRegression::params(3)
        .scale(true)
        .max_iterations(200)
        .algorithm(alg);
    model.fit(&dataset);
}

fn bench(c: &mut Criterion) {
    for (alg, name) in [(Algorithm::Nipals, "Nipals"), (Algorithm::Svd, "Svd")] {
        let mut group = c.benchmark_group("Linfa_pls");
        let sizes: [usize; 3] = [1_000, 10_000, 100_000];
        
        let feat_distr = Laplace::new(0.5, 5.).unwrap();
        let target_distr = DiscreteUniform::new(0, 5).unwrap();

        for size in sizes {
            let dataset = make_dataset(size, 5, feat_distr, target_distr);
            let targets = dataset.targets.into_shape((size, 1)).unwrap();
            let features = dataset.records;
            let dataset = Dataset::new(features, targets);
            let input = (dataset, alg);
            group.bench_with_input(BenchmarkId::new(name, size), &input, |b, (dataset, alg)| {
                b.iter(|| perform_pls(dataset, *alg));
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
