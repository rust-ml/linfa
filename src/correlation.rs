//! Correlation analysis for dataset features
//!
//! # Implementations
//!
//! * Pearsons's Correlation Coefficients - linear feature correlation
use std::fmt;

use ndarray::{Array1, ArrayBase, ArrayView2, Axis, Data, Ix2};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

use crate::dataset::{DatasetBase, Targets};
use crate::Float;

/// Calculate the Pearson's Correlation Coefficient (or bivariate correlation)
///
/// The PCC describes the linear correlation between two variables. It is the covariance divided by
/// the product of the standard deviations, therefore essentially a normalised measurement of the
/// covariance and in range (-1, 1). A negative coefficient indicates a negative correlation
/// between both variables.
fn pearson_correlation<F: Float, D: Data<Elem = F>>(data: &ArrayBase<D, Ix2>) -> Array1<F> {
    // number of obserations and features
    let nobservations = data.nrows();
    let nfeatures = data.ncols();

    // center distribution by subtracting mean
    let mean = data.mean_axis(Axis(0)).unwrap();
    let denoised = data - &mean.insert_axis(Axis(1)).t();

    // calculate the covariance matrix
    let covariance = denoised.t().dot(&denoised) / F::from(nobservations - 1).unwrap();
    // calculate the standard deviation vector
    let std_deviation = denoised.var_axis(Axis(0), F::one()).mapv(|x| x.sqrt());

    // we will only save the upper triangular matrix as the diagonal is one and
    // the lower triangular is a mirror of the upper triangular part
    let mut pearson_coeffs = Array1::zeros(nfeatures * (nfeatures - 1) / 2);

    let mut k = 0;
    for i in 0..(nfeatures - 1) {
        for j in (i + 1)..nfeatures {
            // calculate pearson correlation coefficients by normalizing the covariance matrix
            pearson_coeffs[k] = covariance[(i, j)] / std_deviation[i] / std_deviation[j];

            k += 1;
        }
    }

    pearson_coeffs
}

/// Evidence of non-correlation with re-sampling test
///
/// The p-value supports or reject the null hypthesis that two variables are not correlated. A
/// small p-value indicates a strong evidence that two variables are correlated.
fn p_values<F: Float, D: Data<Elem = F>>(
    data: &ArrayBase<D, Ix2>,
    ground: &Array1<F>,
    num_iter: usize,
) -> Array1<F> {
    // transpose element matrix such that we can shuffle columns
    let (n, m) = (data.ncols(), data.nrows());
    let mut flattened = Vec::with_capacity(n * m);
    for i in 0..m {
        for j in 0..n {
            flattened.push(data[(i, j)]);
        }
    }

    let mut p_values = Array1::zeros(n * (n - 1) / 2);
    let mut rng = SmallRng::from_entropy();

    // calculate p-values by shuffling features `num_iter` times
    for _ in 0..num_iter {
        // shuffle all corresponding features
        for j in 0..n {
            flattened[j * m..(j + 1) * m].shuffle(&mut rng);
        }

        // create an ndarray and calculate the PCC for this distribution
        let arr_view = ArrayView2::from_shape((m, n), &flattened).unwrap();
        let correlation = pearson_correlation(&arr_view.t());

        // count the number of times that the re-shuffled distribution has a larger PCC than the
        // original distribution
        let greater = ground
            .iter()
            .zip(correlation.iter())
            .map(|(a, b)| {
                if a.abs() < b.abs() {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect::<Array1<_>>();

        p_values += &greater;
    }

    // divide by the number of iterations to re-scale range
    p_values / F::from(num_iter).unwrap()
}

/// Pearson Correlation Coefficients (or Bivariate Coefficients)
///
/// The PCCs indicate the linear correlation between variables. This type also supports printing
/// the PCC as an upper triangle matrix together with the feature names.
pub struct PearsonCorrelation<F> {
    pearson_coeffs: Array1<F>,
    p_values: Array1<F>,
    feature_names: Vec<String>,
}

impl<F: Float> PearsonCorrelation<F> {
    /// Calculate the Pearson Correlation Coefficients and optionally p-values from dataset
    ///
    /// The PCC describes the linear correlation between two variables. It is the covariance divided by
    /// the product of the standard deviations, therefore essentially a normalised measurement of the
    /// covariance and in range (-1, 1). A negative coefficient indicates a negative correlation
    /// between both variables.
    ///
    /// The p-value supports or reject the null hypthesis that two variables are not correlated. A
    /// small p-value indicates a strong evidence that two variables are correlated.
    ///
    /// # Parameters
    ///
    /// * `dataset`: Data for the correlation analysis
    /// * `num_iter`: optionally number of iterations of the p-value test, if none then no p-value
    /// are calculate
    ///
    /// # Example
    ///
    /// ```
    /// let corr = linfa_datasets::diabetes()
    ///     .pearson_correlation_with_p_value(100);
    ///
    /// println!("{}", corr);
    /// ```
    ///
    /// The output looks like this (the p-value is in brackets behind the PCC):
    ///
    /// ```ignore
    /// age                        +0.17 (0.61) +0.18 (0.62) +0.33 (0.34) +0.26 (0.47) +0.22 (0.54) -0.07 (0.83) +0.20 (0.60) +0.27 (0.54) +0.30 (0.41)
    /// sex                                     +0.09 (0.74) +0.24 (0.59) +0.04 (0.91) +0.14 (0.74) -0.38 (0.28) +0.33 (0.30) +0.15 (0.74) +0.21 (0.58)
    /// body mass index                                      +0.39 (0.20) +0.25 (0.45) +0.26 (0.51) -0.37 (0.31) +0.41 (0.24) +0.45 (0.21) +0.39 (0.21)
    /// blood pressure                                                    +0.24 (0.54) +0.19 (0.56) -0.18 (0.61) +0.26 (0.45) +0.39 (0.20) +0.39 (0.16)
    /// t-cells                                                                        +0.90 (0.00) +0.05 (0.89) +0.54 (0.05) +0.52 (0.10) +0.33 (0.37)
    /// low-density lipoproteins                                                                    -0.20 (0.53) +0.66 (0.04) +0.32 (0.42) +0.29 (0.42)
    /// high-density lipoproteins                                                                                -0.74 (0.02) -0.40 (0.21) -0.27 (0.42)
    /// thyroid stimulating hormone                                                                                           +0.62 (0.04) +0.42 (0.21)
    /// lamotrigine                                                                                                                        +0.47 (0.14)
    /// blood sugar level
    /// ```

    pub fn from_dataset<D: Data<Elem = F>, T: Targets>(
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        num_iter: Option<usize>,
    ) -> Self {
        // calculate pearson coefficients
        let pearson_coeffs = pearson_correlation(&dataset.records());

        // calculate p values
        let p_values = match num_iter {
            Some(num_iter) => p_values(&dataset.records(), &pearson_coeffs, num_iter),
            None => Array1::zeros(0),
        };

        PearsonCorrelation {
            pearson_coeffs,
            p_values,
            feature_names: dataset.feature_names(),
        }
    }

    /// Return the Pearson's Correlation Coefficients
    ///
    /// The coefficients are describing the linear correlation, normalized in range (-1, 1) between
    /// two variables. Because the correlation is commutative and PCC to the same variable is
    /// always perfectly correlated (i.e. 1), this function only returns the upper triangular
    /// matrix with (n-1)*n/2 elements.
    pub fn get_coeffs(&self) -> &Array1<F> {
        &self.pearson_coeffs
    }

    /// Return the p values supporting the null-hypothesis
    ///
    /// This implementation estimates the p value with the permutation test. As null-hypothesis
    /// the non-correlation between two variables is chosen such that the smaller the p-value the
    /// stronger we can reject the null-hypothesis and conclude that they are linearily correlated.
    pub fn get_p_values(&self) -> Option<&Array1<F>> {
        if self.p_values.is_empty() {
            None
        } else {
            Some(&self.p_values)
        }
    }
}

impl<F: Float, D: Data<Elem = F>, T: Targets> DatasetBase<ArrayBase<D, Ix2>, T> {
    /// Calculate the Pearson Correlation Coefficients from a dataset
    ///
    /// The PCC describes the linear correlation between two variables. It is the covariance divided by
    /// the product of the standard deviations, therefore essentially a normalised measurement of the
    /// covariance and in range (-1, 1). A negative coefficient indicates a negative correlation
    /// between both variables.
    ///
    /// # Example
    ///
    /// ```
    /// let corr = linfa_datasets::diabetes()
    ///     .pearson_correlation();
    ///
    /// println!("{}", corr);
    /// ```
    ///
    pub fn pearson_correlation(&self) -> PearsonCorrelation<F> {
        PearsonCorrelation::from_dataset(self, None)
    }

    /// Calculate the Pearson Correlation Coefficients and p-values from the dataset
    ///
    /// The PCC describes the linear correlation between two variables. It is the covariance divided by
    /// the product of the standard deviations, therefore essentially a normalised measurement of the
    /// covariance and in range (-1, 1). A negative coefficient indicates a negative correlation
    /// between both variables.
    ///
    /// The p-value supports or reject the null hypthesis that two variables are not correlated.
    /// The smaller the p-value the stronger is the evidence that two variables are correlated. A
    /// typical threshold is p < 0.05.
    ///
    /// # Parameters
    ///
    /// * `num_iter`: number of iterations of the permutation test to estimate the p-value
    ///
    /// # Example
    ///
    /// ```
    /// let corr = linfa_datasets::diabetes()
    ///     .pearson_correlation_with_p_value(100);
    ///
    /// println!("{}", corr);
    /// ```
    ///
    pub fn pearson_correlation_with_p_value(&self, num_iter: usize) -> PearsonCorrelation<F> {
        PearsonCorrelation::from_dataset(self, Some(num_iter))
    }
}

/// Display the Pearson's Correlation Coefficients as upper triangular matrix
///
/// This function prints the feature names for each row, the corresponding PCCs and optionally the
/// p-values in brackets behind the PCC.
impl<F: Float> fmt::Display for PearsonCorrelation<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.feature_names.len();
        let longest = self.feature_names.iter().map(|x| x.len()).max().unwrap();

        let mut k = 0;
        for i in 0..(n - 1) {
            write!(f, "{}", self.feature_names[i])?;
            for _ in 0..longest - self.feature_names[i].len() {
                write!(f, " ")?;
            }

            for _ in 0..i {
                write!(f, "             ")?;
            }

            for _ in (i + 1)..n {
                if !self.p_values.is_empty() {
                    write!(
                        f,
                        "{:+.2} ({:.2}) ",
                        self.pearson_coeffs[k], self.p_values[k]
                    )?;
                } else {
                    write!(f, "{:.2} ", self.pearson_coeffs[k])?;
                }

                k += 1;
            }
            writeln!(f,)?;
        }
        writeln!(f, "{}", self.feature_names[n - 1])?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::DatasetBase;
    use ndarray::{stack, Array, Axis};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn uniform_random() {
        let data = Array::random((1000, 4), Uniform::new(-1., 1.));

        let pcc = DatasetBase::from(data).pearson_correlation();

        assert!(pcc.get_coeffs().mapv(|x: f32| x.abs()).sum() < 5e-2 * 6.0);
    }

    #[test]
    fn perfectly_correlated() {
        let v = Array::random((4, 1), Uniform::new(0., 1.));

        // project feature with matrix
        let data = Array::random((1000, 1), Uniform::new(-1., 1.));
        let data_proj = data.dot(&v.t());

        let corr = DatasetBase::from(stack![Axis(1), data, data_proj])
            .pearson_correlation_with_p_value(100);

        assert!(corr.get_coeffs().mapv(|x| 1. - x).sum() < 1e-2);
        assert!(corr.get_p_values().unwrap().sum() < 1e-2);
    }
}
