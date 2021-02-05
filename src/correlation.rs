///! Correlation analysis for dataset features
///
use std::fmt;

use ndarray::{Array1, ArrayBase, Axis, Data, Ix2, ArrayView2};
use rand::{seq::SliceRandom, rngs::SmallRng, SeedableRng};

use crate::dataset::{DatasetBase, Targets};
use crate::Float;

fn pearson_correlation<F: Float, D: Data<Elem = F>>(data: &ArrayBase<D, Ix2>) -> Array1<F> {
    let nobservations = data.nrows();
    let nfeatures = data.ncols();
    
    // subtract mean
    let mean = data.mean_axis(Axis(0)).unwrap();
    let denoised = data - &mean.insert_axis(Axis(1)).t();
    let covariance = denoised.t().dot(&denoised) / F::from(nobservations - 1).unwrap();
    let std_deviation = denoised.var_axis(Axis(0), F::one())
        .mapv(|x| x.sqrt());
    
    let mut pearson_coeffs = Array1::zeros(nfeatures * (nfeatures - 1) / 2);
    
    let mut k = 0;
    for i in 0..(nfeatures-1) {
        for j in (i+1)..nfeatures {
            pearson_coeffs[k] = covariance[(i, j)] / std_deviation[i] / std_deviation[j];

            k += 1;
        }
    }

    pearson_coeffs
}

fn p_values<F: Float, D: Data<Elem = F>>(data: &ArrayBase<D, Ix2>, ground: &Array1<F>, num_iter: usize) -> Array1<F> {
    // transpose element matrix such that we can shuffle columns
    let (n, m) = (data.ncols(), data.nrows());
    let mut flattened = Vec::with_capacity(n * m);
    for i in 0..m {
        for j in 0..n {
            flattened.push(data[(i, j)]);
        }
    }

    let mut p_values = Array1::zeros(n * (n-1) / 2);
    let mut rng = SmallRng::from_entropy();

    // calculate p-values by shuffling features `num_iter` times
    for _ in 0..num_iter {
        // shuffle all corresponding features
        for j in 0..n {
            flattened[j * m .. (j+1)*m].shuffle(&mut rng);
        }

        let arr_view = ArrayView2::from_shape((m, n), &flattened).unwrap();
        let correlation = pearson_correlation(&arr_view.t());

        let greater = ground.iter().zip(correlation.iter())
            .map(|(a, b)| if a.abs() >= b.abs() { F::one() } else { F::zero() })
            .collect::<Array1<_>>();

        p_values += &greater;

        /*
                    for f in 0..n {
            // shuffle all corresponding features
            for j in f+1..n {
                flattened[j * n .. (j+1)*n].shuffle(&mut rng);
            }

            // create a array view
            let arr_view = ArrayView2::from_shape((m, (n-f)), &flattened).unwrap();
            let correlation = pearson_correlation(&arr_view.t());

            let greater = ground.iter().zip(correlation.iter())
                .map(|(a, b)| if a > b { F::one() } else { F::zero() })
                .collect::<Array1<_>>();

            p_values += &greater;
        }*/
    }

    p_values / F::from(num_iter).unwrap()
}

pub struct Correlation<F> {
    pearson_coeffs: Array1<F>,
    p_values: Array1<F>,
    feature_names: Vec<String>,
}

impl<F: Float> Correlation<F> {
    pub fn from_dataset<D: Data<Elem = F>, T: Targets>(
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        p: Option<usize>,
    ) -> Self {
        // calculate pearson coefficients
        let pearson_coeffs = pearson_correlation(&dataset.records());

        // calculate p values
        let p_values = match p {
            Some(p) => p_values(&dataset.records(), &pearson_coeffs, p),
            None => Array1::zeros(0)
        };

        Correlation {
            pearson_coeffs,
            p_values,
            feature_names: dataset.feature_names(),
        }
    }

    pub fn get_coeffs(&self) -> &Array1<F> {
        &self.pearson_coeffs
    }
}

impl<F: Float> fmt::Display for Correlation<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.feature_names.len();
        let longest = self.feature_names.iter().map(|x| x.len()).max()
            .unwrap();

        let mut k = 0;
        for i in 0..(n-1) {
            write!(f, "{}", self.feature_names[i])?;
            for _ in 0..longest - self.feature_names[i].len() {
                write!(f, " ")?;
            }

            for _ in 0..i {
                write!(f, "            ")?;
            }

            for _ in (i+1)..n {
                if self.p_values.len() > 0 {
                    write!(f, "{:.2} ({:.2}) ", self.pearson_coeffs[k], self.p_values[k])?;
                } else {
                    write!(f, "{:.2} ", self.pearson_coeffs[k])?;
                }

                k += 1;
            }
            write!(f, "\n")?;
        }
        write!(f, "{}\n", self.feature_names[n-1])?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use super::{Correlation, DatasetBase};

    #[test]
    fn correlation_random() {
        let data = Array::random((1000, 4), Uniform::new(-1., 1.));

        let dataset = DatasetBase::from(data);

        let corr = Correlation::from_dataset(&dataset, None);

        assert!(corr.get_coeffs().mapv(|x: f32| x.abs()).sum() < 5e-2 * 6.0);
    }

    #[test]
    fn correlation_diabetes() {
        let d = linfa_datasets::diabetes();
        let dataset = DatasetBase::new(d.records().to_owned(), ())
            .with_feature_names(d.feature_names());

        let corr = Correlation::from_dataset(&dataset, Some(1000));

        println!("{}", corr);
    }
}
