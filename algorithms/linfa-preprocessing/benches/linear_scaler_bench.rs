use linfa::traits::Fit;
use linfa::traits::Transformer;
use linfa_preprocessing::linear_scaling::LinearScaler;
use ndarray::Array2;
use ndarray_rand::{
    rand::distributions::Uniform, rand::rngs::SmallRng, rand::SeedableRng, RandomExt,
};

fn standard_scaler_bench() {
    let mut rng = SmallRng::seed_from_u64(42);
    for size in (1000..10000).step_by(1000) {
        fit_transform_scaler(LinearScaler::standard(), &mut rng, size);
    }
}

fn min_max_scaler_bench() {
    let mut rng = SmallRng::seed_from_u64(42);
    for size in (1000..10000).step_by(1000) {
        fit_transform_scaler(LinearScaler::min_max(), &mut rng, size);
    }
}

fn max_abs_scaler_bench() {
    let mut rng = SmallRng::seed_from_u64(42);
    for size in (1000..10000).step_by(1000) {
        fit_transform_scaler(LinearScaler::max_abs(), &mut rng, size);
    }
}

fn fit_transform_scaler(scaler: LinearScaler<f64>, rng: &mut SmallRng, size: usize) {
    let dataset = Array2::random_using((size, 7), Uniform::from(-30. ..30.), rng).into();
    scaler
        .fit(iai::black_box(&dataset))
        .unwrap()
        .transform(iai::black_box(dataset));
}

iai::main!(
    standard_scaler_bench,
    min_max_scaler_bench,
    max_abs_scaler_bench
);
