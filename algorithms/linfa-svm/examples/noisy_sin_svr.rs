use linfa::prelude::*;
use linfa_svm::{error::Result, Svm};
use ndarray::Array1;
use ndarray_rand::{
    rand::{Rng, SeedableRng},
    rand_distr::Uniform,
};
use rand_xoshiro::Xoshiro256Plus;

/// Example inspired by https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
fn main() -> Result<()> {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    let range = Uniform::new(0., 5.);
    let mut x: Vec<f64> = (0..40).map(|_| rng.sample(range)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let x = Array1::from_vec(x);

    let mut y = x.mapv(|v| v.sin());

    // add some noise
    y.iter_mut()
        .enumerate()
        .filter(|(i, _)| i % 5 == 0)
        .for_each(|(_, y)| *y = 3. * (0.5 - rng.gen::<f64>()));

    let x = x.into_shape((40, 1)).unwrap();
    let dataset = DatasetBase::new(x, y);
    let model = Svm::params()
        .c_svr(100., Some(0.1))
        .gaussian_kernel(10.)
        .fit(&dataset)?;

    println!("{model}");

    let predicted = model.predict(&dataset);
    let err = predicted.mean_squared_error(&dataset).unwrap();
    println!("err={err}");

    Ok(())
}
