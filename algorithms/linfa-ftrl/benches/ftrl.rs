use linfa::prelude::Predict;
use linfa::traits::FitWith;
use linfa::{Dataset, DatasetBase, ParamGuard};
use linfa_ftrl::FTRL;
use ndarray::{Array1, Array2};
use ndarray_rand::{
    rand::distributions::Uniform, rand::rngs::SmallRng, rand::SeedableRng, RandomExt,
};

fn iai_fit_and_predict_without_prior_model() {
    let mut rng = SmallRng::seed_from_u64(42);
    let params = FTRL::params();
    for nfeatures in (10..100).step_by(10) {
        for size in (1_000..10_000).step_by(2_000) {
            let dataset = get_dataset(&mut rng, size, nfeatures);
            let model = params.fit_with(None, iai::black_box(&dataset)).unwrap();
            model.predict(iai::black_box(dataset));
        }
    }
}

fn fit_and_predict_with_prior_model() {
    let mut rng = SmallRng::seed_from_u64(42);
    let valid_params = FTRL::params().check().unwrap();
    let params = FTRL::params();
    for nfeatures in (10..100).step_by(10) {
        for size in (1_000..10_000).step_by(2_000) {
            let mut model = FTRL::new(valid_params.clone(), nfeatures);
            let dataset = get_dataset(&mut rng, size, nfeatures);
            model = params
                .fit_with(iai::black_box(Some(model)), iai::black_box(&dataset))
                .unwrap();
            model.predict(iai::black_box(dataset));
        }
    }
}

fn to_binary(value: f32) -> bool {
    value >= 0.5
}

fn get_dataset(
    rng: &mut SmallRng,
    size: usize,
    nfeatures: usize,
) -> DatasetBase<Array2<f64>, Array1<bool>> {
    let features = Array2::random_using((size, nfeatures), Uniform::from(-30. ..30.), rng);
    let target = Array1::random_using(size, Uniform::from(0. ..1.), rng).mapv(to_binary);
    Dataset::new(features, target)
}

iai::main!(
    iai_fit_and_predict_without_prior_model,
    fit_and_predict_with_prior_model
);
