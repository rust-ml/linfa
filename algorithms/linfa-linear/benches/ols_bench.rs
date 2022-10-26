use iai::black_box;
use linfa::dataset::Dataset;
use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use ndarray::{Array, Ix1};
use std::error::Error;

fn create_dataset(sample_size: usize) -> Dataset<f64, f64, Ix1> {
    let num_cols: usize = 5;

    let array = Array::from_elem((sample_size, num_cols), 7.);
    let targets = Array::from_elem(sample_size, 7.);
    Dataset::new(array, targets)
}

fn iai_ols_1_000_bench() -> Result<(), Box<dyn Error>> {
    let dataset = create_dataset(1_000);
    let lin_reg = LinearRegression::new();
    lin_reg.fit(&black_box(dataset))?;
    Ok(())
}

fn iai_ols_10_000_bench() -> Result<(), Box<dyn Error>> {
    let dataset = create_dataset(10_000);
    let lin_reg = LinearRegression::new();
    lin_reg.fit(&black_box(dataset))?;
    Ok(())
}

fn iai_ols_100_000_bench() -> Result<(), Box<dyn Error>> {
    let dataset = create_dataset(100_000);
    let lin_reg = LinearRegression::new();
    lin_reg.fit(&black_box(dataset))?;
    Ok(())
}

iai::main!(
    iai_ols_1_000_bench,
    iai_ols_10_000_bench,
    iai_ols_100_000_bench
);
