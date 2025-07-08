use linfa::prelude::*;
use linfa_pls::{PlsRegression, Result};
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand_xoshiro::Xoshiro256Plus;

#[allow(clippy::many_single_char_names)]
fn main() -> Result<()> {
    let n = 1000;
    let q = 3;
    let p = 10;
    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    // X shape (n, p) random
    let x: Array2<f64> = Array::random_using((n, p), StandardNormal, &mut rng);

    // B shape (p, q) such that B[0, ..] = 1, B[1, ..] = 2; otherwise zero
    let mut b: Array2<f64> = Array2::zeros((p, q));
    b.row_mut(0).assign(&Array1::ones(q));
    b.row_mut(1).assign(&Array1::from_elem(q, 2.));

    // Y shape (n, q) such that yj = 1*x1 + 2*x2 + noise(Normal(5, 1))
    let y = x.dot(&b) + Array::random_using((n, q), StandardNormal, &mut rng).mapv(|v: f64| v + 5.);

    let ds = Dataset::new(x, y);
    let pls = PlsRegression::params(3)
        .scale(true)
        .max_iterations(200)
        .fit(&ds)?;

    println!("True B (such that: Y = XB + noise)");
    println!("{b:?}");

    // PLS regression coefficients is an estimation of B
    println!("Estimated B");
    println!("{:1.1}", pls.coefficients());
    Ok(())
}
