///! Correlation analysis for dataset features
///
use ndarray::{Array1, ArrayBase, Axis, Data, Ix2};

use crate::dataset::{DatasetBase, Records, Targets};
use crate::Float;

pub struct Correlation<F> {
    pearson_coeffs: Array1<F>,
    p_values: Array1<F>,
}

impl<F: Float> Correlation<F> {
    pub fn from_dataset<D: Data<Elem = F>, T: Targets>(
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Self {
        let nobs = dataset.observations();

        // subtract mean
        let mean = dataset.records().mean_axis(Axis(0)).unwrap();
        let denoised = dataset.records() - &mean.insert_axis(Axis(1));
        let covariance = denoised.dot(&denoised.t()) / F::from(nobs - 1).unwrap();
        let variance = denoised.var_axis(Axis(0), F::one());

        let mut pearson_coeffs = Array1::zeros(nobs * (nobs - 1) / 2);

        for i in 0..nobs {
            for j in 0..i {
                pearson_coeffs[j + i * nobs] = covariance[(i, j)] / variance[i] / variance[j];
            }
        }

        Correlation {
            pearson_coeffs,
            p_values: Array1::zeros(0),
        }
    }
}
