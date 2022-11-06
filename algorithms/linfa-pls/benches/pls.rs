use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::traits::Fit;
use linfa_datasets::generate::make_dataset;
use linfa_pls::PlsRegression;
use linfa::Dataset;
use statrs::distribution::{DiscreteUniform, Laplace};

#[allow(unused_must_use)]
fn perform_pls(num_rows: usize) {
    let feat_distr = Laplace::new(0.5, 5.).unwrap();
    let target_distr = DiscreteUniform::new(0, 5).unwrap();
    let num_feats: usize = 5;
    let dataset = make_dataset(num_rows, num_feats, feat_distr, target_distr);
    let targets = dataset.targets.into_shape((num_rows, 1)).unwrap();
    let features = dataset.records;
    let dataset = Dataset::new(features, targets);
    let model = PlsRegression::params(3)
        .scale(true)
        .max_iterations(200);
    model.fit(&dataset);
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linfa_pls");
    let sizes: [usize; 3] = [1_000, 10_000, 100_000];
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("SVD", size), &size, |b, size| {
            b.iter(|| perform_pls(*size));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
