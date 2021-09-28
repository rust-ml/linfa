use linfa::traits::{Fit, Transformer};
use linfa_preprocessing::linear_scaling::{LinearScaler, LinearScalerParams};
use ndarray::Array2;
use ndarray_rand::{
    rand::distributions::Uniform, rand::rngs::SmallRng, rand::SeedableRng, RandomExt,
};

fn iai_standard_scaler_bench() {
    let mut rng = SmallRng::seed_from_u64(42);
    for nfeatures in (10..100).step_by(10) {
        fit_transform_scaler(LinearScaler::standard(), &mut rng, 10000, nfeatures);
    }
}

fn iai_min_max_scaler_bench() {
    let mut rng = SmallRng::seed_from_u64(42);
    for nfeatures in (10..100).step_by(10) {
        fit_transform_scaler(LinearScaler::min_max(), &mut rng, 10000, nfeatures);
    }
}

fn iai_max_abs_scaler_bench() {
    let mut rng = SmallRng::seed_from_u64(42);
    for nfeatures in (10..100).step_by(10) {
        fit_transform_scaler(LinearScaler::max_abs(), &mut rng, 10000, nfeatures);
    }
}

fn fit_transform_scaler(
    scaler: LinearScalerParams<f64>,
    rng: &mut SmallRng,
    size: usize,
    nfeatures: usize,
) {
    let dataset = Array2::random_using((size, nfeatures), Uniform::from(-30. ..30.), rng).into();
    scaler
        .fit(iai::black_box(&dataset))
        .unwrap()
        .transform(iai::black_box(dataset));
}

iai::main!(
    iai_standard_scaler_bench,
    iai_min_max_scaler_bench,
    iai_max_abs_scaler_bench
);
