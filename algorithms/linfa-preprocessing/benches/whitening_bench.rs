use linfa::traits::Fit;
use linfa::traits::Transformer;
use linfa_preprocessing::whitening::Whitener;
use ndarray::Array2;
use ndarray_rand::{
    rand::distributions::Uniform, rand::rngs::SmallRng, rand::SeedableRng, RandomExt,
};

fn pca_bench() {
    let mut rng = SmallRng::seed_from_u64(21);
    for size in (1000..10000).step_by(1000) {
        fit_transform_whitener(Whitener::pca(), &mut rng, size);
    }
}

fn zca_bench() {
    let mut rng = SmallRng::seed_from_u64(21);
    for size in (1000..10000).step_by(1000) {
        fit_transform_whitener(Whitener::zca(), &mut rng, size);
    }
}

fn cholesky_bench() {
    let mut rng = SmallRng::seed_from_u64(21);
    for size in (1000..10000).step_by(1000) {
        fit_transform_whitener(Whitener::cholesky(), &mut rng, size);
    }
}

fn fit_transform_whitener(whitener: Whitener, rng: &mut SmallRng, size: usize) {
    let dataset = Array2::random_using((size, 7), Uniform::from(-30. ..30.), rng).into();
    whitener
        .fit(iai::black_box(&dataset))
        .unwrap()
        .transform(iai::black_box(dataset));
}

iai::main!(pca_bench, zca_bench, cholesky_bench);
