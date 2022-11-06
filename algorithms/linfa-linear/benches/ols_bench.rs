use statrs::distribution::{DiscreteUniform, Laplace};
use linfa_datasets::generate::make_dataset;
use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn perform_ols(num_rows: usize) {
    let feat_distr = Laplace::new(0.5, 5. ).unwrap();
    let target_distr = DiscreteUniform::new(0, 5).unwrap();
    let num_feats: usize = 5;
    let dataset = make_dataset(num_rows, num_feats, feat_distr, target_distr);
    let lin_reg = LinearRegression::new();
    let _model = lin_reg.fit(&dataset);
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linfa_linear");
    let sizes: [usize; 3] = [1_000, 10_000, 100_000];
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("OLS", size), &size, |b, size| {
            b.iter(|| perform_ols(*size));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
