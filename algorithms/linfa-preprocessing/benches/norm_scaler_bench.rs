use linfa::traits::Transformer;
use linfa_preprocessing::norm_scaling::NormScaler;
use ndarray::Array2;
use ndarray_rand::{
    rand::distributions::Uniform, rand::rngs::SmallRng, rand::SeedableRng, RandomExt,
};

fn iai_l2_scaler_bench() {
    let mut rng = SmallRng::seed_from_u64(84);
    for nfeatures in (10..100).step_by(10) {
        transform_scaler(NormScaler::l2(), &mut rng, 10000, nfeatures);
    }
}

fn iai_l1_scaler_bench() {
    let mut rng = SmallRng::seed_from_u64(84);
    for nfeatures in (10..100).step_by(10) {
        transform_scaler(NormScaler::l1(), &mut rng, 10000, nfeatures);
    }
}

fn iai_max_scaler_bench() {
    let mut rng = SmallRng::seed_from_u64(84);
    for nfeatures in (10..100).step_by(10) {
        transform_scaler(NormScaler::max(), &mut rng, 10000, nfeatures);
    }
}

fn transform_scaler(scaler: NormScaler, rng: &mut SmallRng, size: usize, nfeatures: usize) {
    let dataset: Array2<f64> =
        Array2::random_using((size, nfeatures), Uniform::from(-30. ..30.), rng).into();
    scaler.transform(iai::black_box(dataset));
}

iai::main!(
    iai_l2_scaler_bench,
    iai_l1_scaler_bench,
    iai_max_scaler_bench
);
