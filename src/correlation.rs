///! Correlation analysis for dataset features
///
use ndarray::{Array1, ArrayBase, Axis, Data, Ix2};

use crate::dataset::{DatasetBase, Records, Targets};
use crate::Float;

fn pearson_correlation<F: Float, D: Data<Elem = F>>(data: &ArrayBase<D, Ix2>) -> Array1<F> {
    let nobs = data.len_of(Axis(1));
    
    // subtract mean
    let mean = data.mean_axis(Axis(0)).unwrap();
    let denoised = data - &mean.insert_axis(Axis(1));
    let covariance = denoised.dot(&denoised.t()) / F::from(nobs - 1).unwrap();
    let variance = denoised.var_axis(Axis(0), F::one());
    
    let mut pearson_coeffs = Array1::zeros(nobs * (nobs - 1) / 2);
    
    for i in 1..nobs {
        for j in i..nobs {
            pearson_coeffs[j + i * nobs] = covariance[(i, j)] / variance[i] / variance[j];
        }
    }

    pearson_coeffs
}

fn p_values<F: Float, D: Data<Elem = F>>(data: &ArrayBase<D, Ix2>, ground: &Array1<F>, p: f32, num_iter: usize) -> Array1<F> {
    // transpose element matrix such that we can shuffle columns
    let [n, m] = data.shape();
    let mut flattened = Vec::with_capacity(n * m);
    for i in 0..*n {
        for j in 0..*m {
            flattened.push(data[(i, j)]);
        }
    }

    let mut p_values = Array1::zeros(m * (m-1) / 2);

    // calculate p-values by shuffling features `num_iter` times
    for i in 0..num_iter {
        for f in 0..n {
            // shuffle all corresponding features
            for j in f..n {
                flattened[j * n .. (j+1)*n].shuffle();
            }
        }

    }

    dbg!(&data);

    Array1::zeros(0)
}

pub struct Correlation<F> {
    pearson_coeffs: Array1<F>,
    p_values: Array1<F>,
}

impl<F: Float> Correlation<F> {
    pub fn from_dataset<D: Data<Elem = F>, T: Targets>(
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        p: Option<f32>,
    ) -> Self {
        // calculate pearson coefficients
        let pearson_coeffs = pearson_correlation(&dataset.records());

        // calculate p values
        let p_values = match p {
            Some(p) => p_values(&dataset.records(), p),
            None => Array1::zeros(0)
        };

        Correlation {
            pearson_coeffs,
            p_values
        }
    }
}
