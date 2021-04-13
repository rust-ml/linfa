use linfa::traits::Fit;
use linfa::traits::Transformer;
use linfa_preprocessing::whitening::Whitener;
use ndarray::Array2;
use ndarray_rand::{
    rand::distributions::Uniform, rand::rngs::SmallRng, rand::SeedableRng, RandomExt,
};

fn iai_pca_bench() {
    let mut rng = SmallRng::seed_from_u64(21);
    for nfeatures in (10..100).step_by(10) {
        fit_transform_whitener(Whitener::pca(), &mut rng, 10000, nfeatures);
    }
}

fn iai_zca_bench() {
    let mut rng = SmallRng::seed_from_u64(21);
    for nfeatures in (10..100).step_by(10) {
        fit_transform_whitener(Whitener::zca(), &mut rng, 10000, nfeatures);
    }
}

fn iai_cholesky_bench() {
    let mut rng = SmallRng::seed_from_u64(21);
    for nfeatures in (10..100).step_by(10) {
        fit_transform_whitener(Whitener::cholesky(), &mut rng, 10000, nfeatures);
    }
}

fn fit_transform_whitener(whitener: Whitener, rng: &mut SmallRng, size: usize, nfeatures: usize) {
    let dataset = Array2::random_using((size, nfeatures), Uniform::from(-30. ..30.), rng).into();
    whitener
        .fit(iai::black_box(&dataset))
        .unwrap()
        .transform(iai::black_box(dataset));
}

iai::main!(iai_pca_bench, iai_zca_bench, iai_cholesky_bench);
