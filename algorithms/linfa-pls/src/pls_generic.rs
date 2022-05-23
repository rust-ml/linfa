use crate::errors::{PlsError, Result};
use crate::utils;
use crate::{PlsParams, PlsValidParams};

use linfa::{
    dataset::{Records, WithLapack, WithoutLapack},
    traits::Fit,
    traits::PredictInplace,
    traits::Transformer,
    Dataset, DatasetBase, Float,
};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::svd::*;
use ndarray_stats::QuantileExt;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Pls<F: Float> {
    x_mean: Array1<F>,
    x_std: Array1<F>,
    y_mean: Array1<F>,
    y_std: Array1<F>,
    x_weights: Array2<F>,  // U
    y_weights: Array2<F>,  // V
    x_scores: Array2<F>,   // xi
    y_scores: Array2<F>,   // Omega
    x_loadings: Array2<F>, // Gamma
    y_loadings: Array2<F>, // Delta
    x_rotations: Array2<F>,
    y_rotations: Array2<F>,
    coefficients: Array2<F>,
    n_iters: Array1<usize>,
}

#[derive(PartialEq, Debug, Clone, Copy, Eq, Hash)]
pub enum Algorithm {
    Nipals,
    Svd,
}

#[derive(PartialEq, Debug, Clone, Copy, Eq, Hash)]
pub(crate) enum DeflationMode {
    Regression,
    Canonical,
}

#[derive(PartialEq, Debug, Clone, Copy, Eq, Hash)]
pub(crate) enum Mode {
    A,
    B,
}

/// Generic PLS algorithm.
/// Main ref: Wegelin, a survey of Partial Least Squares (PLS) methods,
/// with emphasis on the two-block case
/// https://www.stat.washington.edu/research/reports/2000/tr371.pdf
impl<F: Float> Pls<F> {
    // Constructor for PlsRegression method
    pub fn regression(n_components: usize) -> PlsParams<F> {
        PlsParams::new(n_components)
    }

    // Constructor for PlsCanonical method
    pub fn canonical(n_components: usize) -> PlsParams<F> {
        PlsParams::new(n_components).deflation_mode(DeflationMode::Canonical)
    }

    // Constructor for PlsCca method
    pub fn cca(n_components: usize) -> PlsParams<F> {
        PlsParams::new(n_components)
            .deflation_mode(DeflationMode::Canonical)
            .mode(Mode::B)
    }

    pub fn weights(&self) -> (&Array2<F>, &Array2<F>) {
        (&self.x_weights, &self.y_weights)
    }

    #[cfg(test)]
    pub fn scores(&self) -> (&Array2<F>, &Array2<F>) {
        (&self.x_scores, &self.y_scores)
    }

    pub fn loadings(&self) -> (&Array2<F>, &Array2<F>) {
        (&self.x_loadings, &self.y_loadings)
    }

    pub fn rotations(&self) -> (&Array2<F>, &Array2<F>) {
        (&self.x_rotations, &self.y_rotations)
    }

    pub fn coefficients(&self) -> &Array2<F> {
        &self.coefficients
    }

    pub fn inverse_transform(
        &self,
        dataset: DatasetBase<
            ArrayBase<impl Data<Elem = F>, Ix2>,
            ArrayBase<impl Data<Elem = F>, Ix2>,
        >,
    ) -> DatasetBase<Array2<F>, Array2<F>> {
        let mut x_orig = dataset.records().dot(&self.x_loadings.t());
        x_orig = &x_orig * &self.x_std;
        x_orig = &x_orig + &self.x_mean;
        let mut y_orig = dataset.targets().dot(&self.y_loadings.t());
        y_orig = &y_orig * &self.y_std;
        y_orig = &y_orig + &self.y_mean;
        Dataset::new(x_orig, y_orig)
    }
}

impl<F: Float, D: Data<Elem = F>>
    Transformer<
        DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        DatasetBase<Array2<F>, Array2<F>>,
    > for Pls<F>
{
    fn transform(
        &self,
        dataset: DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> DatasetBase<Array2<F>, Array2<F>> {
        let mut x_norm = dataset.records() - &self.x_mean;
        x_norm /= &self.x_std;
        let mut y_norm = dataset.targets() - &self.y_mean;
        y_norm /= &self.y_std;
        // Apply rotations
        let x_proj = x_norm.dot(&self.x_rotations);
        let y_proj = y_norm.dot(&self.y_rotations);
        Dataset::new(x_proj, y_proj)
    }
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array2<F>> for Pls<F> {
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array2<F>) {
        assert_eq!(
            y.shape(),
            &[x.nrows(), self.coefficients.ncols()],
            "The number of data points must match the number of output targets."
        );

        let mut x = x - &self.x_mean;
        x /= &self.x_std;
        *y = x.dot(&self.coefficients) + &self.y_mean;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
        Array2::zeros((x.nrows(), self.coefficients.ncols()))
    }
}

impl<F: Float, D: Data<Elem = F>> Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, PlsError>
    for PlsValidParams<F>
{
    type Object = Pls<F>;

    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        let records = dataset.records();
        let targets = dataset.targets();

        let n = records.nrows();
        let p = records.ncols();
        let q = targets.ncols();

        if n < 2 {
            return Err(PlsError::NotEnoughSamplesError(
                dataset.records().nsamples(),
            ));
        }

        let n_components = self.n_components();
        let rank_upper_bound = match self.deflation_mode() {
            DeflationMode::Regression => {
                // With PLSRegression n_components is bounded by the rank of (x.T x)
                // see Wegelin page 25
                p
            }
            DeflationMode::Canonical => {
                // With CCA and PLSCanonical, n_components is bounded by the rank of
                // X and the rank of Y: see Wegelin page 12
                n.min(p.min(q))
            }
        };

        if 1 > n_components || n_components > rank_upper_bound {
            return Err(PlsError::BadComponentNumberError {
                upperbound: rank_upper_bound,
                actual: n_components,
            });
        }
        let norm_y_weights = self.deflation_mode() == DeflationMode::Canonical;
        let (mut xk, mut yk, x_mean, y_mean, x_std, y_std) =
            utils::center_scale_dataset(dataset, self.scale());

        let mut x_weights = Array2::<F>::zeros((p, n_components)); // U
        let mut y_weights = Array2::<F>::zeros((q, n_components)); // V
        let mut x_scores = Array2::<F>::zeros((n, n_components)); // xi
        let mut y_scores = Array2::<F>::zeros((n, n_components)); // Omega
        let mut x_loadings = Array2::<F>::zeros((p, n_components)); // Gamma
        let mut y_loadings = Array2::<F>::zeros((q, n_components)); // Delta
        let mut n_iters = Array1::zeros(n_components);

        // This whole thing corresponds to the algorithm in section 4.1 of the
        // review from Wegelin. See above for a notation mapping from code to
        // paper.
        let eps = F::epsilon();
        for k in 0..n_components {
            // Find first left and right singular vectors of the x.T.dot(Y)
            // cross-covariance matrix.

            let (mut x_weights_k, mut y_weights_k) = match self.algorithm() {
                Algorithm::Nipals => {
                    // Replace columns that are all close to zero with zeros
                    for mut yj in yk.columns_mut() {
                        if *(yj.mapv(|y| y.abs()).max()?) < F::cast(10.) * eps {
                            yj.assign(&Array1::zeros(yj.len()));
                        }
                    }

                    let (x_weights_k, y_weights_k, n_iter) =
                        self.get_first_singular_vectors_power_method(&xk, &yk, norm_y_weights)?;
                    n_iters[k] = n_iter;
                    (x_weights_k, y_weights_k)
                }
                Algorithm::Svd => self.get_first_singular_vectors_svd(&xk, &yk)?,
            };
            utils::svd_flip_1d(&mut x_weights_k, &mut y_weights_k);

            // compute scores, i.e. the projections of x and Y
            let x_scores_k = xk.dot(&x_weights_k);
            let y_ss = if norm_y_weights {
                F::one()
            } else {
                y_weights_k.dot(&y_weights_k)
            };
            let y_scores_k = yk.dot(&y_weights_k) / y_ss;

            // Deflation: subtract rank-one approx to obtain xk+1 and yk+1
            let x_loadings_k = x_scores_k.dot(&xk) / x_scores_k.dot(&x_scores_k);
            xk = xk - utils::outer(&x_scores_k, &x_loadings_k); // outer product

            let y_loadings_k = match self.deflation_mode() {
                DeflationMode::Canonical => {
                    // regress yk on y_score
                    let y_loadings_k = y_scores_k.dot(&yk) / y_scores_k.dot(&y_scores_k);
                    yk = yk - utils::outer(&y_scores_k, &y_loadings_k); // outer product
                    y_loadings_k
                }
                DeflationMode::Regression => {
                    // regress yk on x_score
                    let y_loadings_k = x_scores_k.dot(&yk) / x_scores_k.dot(&x_scores_k);
                    yk = yk - utils::outer(&x_scores_k, &y_loadings_k); // outer product
                    y_loadings_k
                }
            };

            x_weights.column_mut(k).assign(&x_weights_k);
            y_weights.column_mut(k).assign(&y_weights_k);
            x_scores.column_mut(k).assign(&x_scores_k);
            y_scores.column_mut(k).assign(&y_scores_k);
            x_loadings.column_mut(k).assign(&x_loadings_k);
            y_loadings.column_mut(k).assign(&y_loadings_k);
        }
        // x was approximated as xi . Gamma.T + x_(R+1) xi . Gamma.T is a sum
        // of n_components rank-1 matrices. x_(R+1) is whatever is left
        // to fully reconstruct x, and can be 0 if x is of rank n_components.
        // Similiarly, Y was approximated as Omega . Delta.T + Y_(R+1)

        // Compute transformation matrices (rotations_). See User Guide.
        let x_rotations = x_weights.dot(&utils::pinv2(x_loadings.t().dot(&x_weights).view(), None));
        let y_rotations = y_weights.dot(&utils::pinv2(y_loadings.t().dot(&y_weights).view(), None));

        let mut coefficients = x_rotations.dot(&y_loadings.t());
        coefficients *= &y_std;

        Ok(Pls {
            x_mean,
            x_std,
            y_mean,
            y_std,
            x_weights,
            y_weights,
            x_scores,
            y_scores,
            x_loadings,
            y_loadings,
            x_rotations,
            y_rotations,
            coefficients,
            n_iters,
        })
    }
}

impl<F: Float> PlsValidParams<F> {
    /// Return the first left and right singular vectors of x'Y.
    /// Provides an alternative to the svd(x'Y) and uses the power method instead.
    fn get_first_singular_vectors_power_method(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        y: &ArrayBase<impl Data<Elem = F>, Ix2>,
        norm_y_weights: bool,
    ) -> Result<(Array1<F>, Array1<F>, usize)> {
        let eps = F::epsilon();

        let mut y_score = None;
        for col in y.t().rows() {
            if *col.mapv(|v| v.abs()).max().unwrap() > eps {
                y_score = Some(col.to_owned());
                break;
            }
        }
        let mut y_score = y_score.ok_or(PlsError::PowerMethodConstantResidualError())?;

        let mut x_pinv = None;
        let mut y_pinv = None;
        if self.mode() == Mode::B {
            x_pinv = Some(utils::pinv2(x.view(), Some(F::cast(10.) * eps)));
            y_pinv = Some(utils::pinv2(y.view(), Some(F::cast(10.) * eps)));
        }

        // init to big value for first convergence check
        let mut x_weights_old = Array1::<F>::from_elem(x.ncols(), F::cast(100.));

        let mut n_iter = 1;
        let mut x_weights = Array1::<F>::ones(x.ncols());
        let mut y_weights = Array1::<F>::ones(y.ncols());
        let mut converged = false;
        while n_iter < self.max_iter() {
            x_weights = match self.mode() {
                Mode::A => x.t().dot(&y_score) / y_score.dot(&y_score),
                Mode::B => x_pinv.to_owned().unwrap().dot(&y_score),
            };
            x_weights /= x_weights.dot(&x_weights).sqrt() + eps;
            let x_score = x.dot(&x_weights);

            y_weights = match self.mode() {
                Mode::A => y.t().dot(&x_score) / x_score.dot(&x_score),
                Mode::B => y_pinv.to_owned().unwrap().dot(&x_score),
            };

            if norm_y_weights {
                y_weights /= y_weights.dot(&y_weights).sqrt() + eps
            }

            let ya = y.dot(&y_weights);
            let yb = y_weights.dot(&y_weights) + eps;
            y_score = ya.mapv(|v| v / yb);

            let x_weights_diff = &x_weights - &x_weights_old;
            if x_weights_diff.dot(&x_weights_diff) < self.tolerance() || y.ncols() == 1 {
                converged = true;
                break;
            } else {
                x_weights_old = x_weights.to_owned();
                n_iter += 1;
            }
        }
        if n_iter == self.max_iter() && !converged {
            Err(PlsError::PowerMethodNotConvergedError(self.max_iter()))
        } else {
            Ok((x_weights, y_weights, n_iter))
        }
    }

    fn get_first_singular_vectors_svd(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        y: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> Result<(Array1<F>, Array1<F>)> {
        let c = x.t().dot(y);

        let c = c.with_lapack();
        let (u, _, vt) = c.svd(true, true)?;
        // safe unwrap because both parameters are set to true in above call
        let u = u.unwrap().column(0).to_owned().without_lapack();
        let vt = vt.unwrap().row(0).to_owned().without_lapack();

        Ok((u, vt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::{dataset::Records, traits::Predict, ParamGuard};
    use linfa_datasets::linnerud;
    use ndarray::{array, concatenate, Array, Axis};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<PlsParams<f64>>();
        has_autotraits::<PlsValidParams<f64>>();
        has_autotraits::<Pls<f64>>();
        has_autotraits::<PlsError>();
    }

    fn assert_matrix_orthonormal(m: &Array2<f64>) {
        assert_abs_diff_eq!(&m.t().dot(m), &Array::eye(m.ncols()), epsilon = 1e-7);
    }

    fn assert_matrix_orthogonal(m: &Array2<f64>) {
        let k = m.t().dot(m);
        assert_abs_diff_eq!(&k, &Array::from_diag(&k.diag()), epsilon = 1e-7);
    }

    #[test]
    fn test_pls_canonical_basics() -> Result<()> {
        // Basic checks for PLSCanonical
        let dataset = linnerud();
        let records = dataset.records();

        let pls = Pls::canonical(records.ncols()).fit(&dataset)?;

        let (x_weights, y_weights) = pls.weights();
        assert_matrix_orthonormal(x_weights);
        assert_matrix_orthonormal(y_weights);

        let (x_scores, y_scores) = pls.scores();
        assert_matrix_orthogonal(x_scores);
        assert_matrix_orthogonal(y_scores);

        // Check X = TP' and Y = UQ'
        let (p, q) = pls.loadings();
        let t = x_scores;
        let u = y_scores;

        // Need to scale first
        let (xc, yc, ..) = utils::center_scale_dataset(&dataset, true);
        assert_abs_diff_eq!(&xc, &t.dot(&p.t()), epsilon = 1e-7);
        assert_abs_diff_eq!(&yc, &u.dot(&q.t()), epsilon = 1e-7);

        // Check that rotations on training data lead to scores
        let ds = pls.transform(dataset);
        assert_abs_diff_eq!(ds.records(), x_scores, epsilon = 1e-7);
        assert_abs_diff_eq!(ds.targets(), y_scores, epsilon = 1e-7);

        Ok(())
    }

    #[test]
    fn test_sanity_check_pls_regression() {
        let dataset = linnerud();
        let pls = Pls::regression(3)
            .fit(&dataset)
            .expect("PLS fitting failed");

        // The results were checked against scikit-learn 0.24 PlsRegression
        let expected_x_weights = array![
            [0.61330704, -0.00443647, 0.78983213],
            [0.74697144, -0.32172099, -0.58183269],
            [0.25668686, 0.94682413, -0.19399983]
        ];

        let expected_x_loadings = array![
            [0.61470416, -0.24574278, 0.78983213],
            [0.65625755, -0.14396183, -0.58183269],
            [0.51733059, 1.00609417, -0.19399983]
        ];

        let expected_y_weights = array![
            [-0.32456184, 0.29892183, 0.20316322],
            [-0.42439636, 0.61970543, 0.19320542],
            [0.13143144, -0.26348971, -0.17092916]
        ];

        let expected_y_loadings = array![
            [-0.32456184, 0.29892183, 0.20316322],
            [-0.42439636, 0.61970543, 0.19320542],
            [0.13143144, -0.26348971, -0.17092916]
        ];
        assert_abs_diff_eq!(pls.x_weights, expected_x_weights, epsilon = 1e-6);
        assert_abs_diff_eq!(pls.x_loadings, expected_x_loadings, epsilon = 1e-6);
        assert_abs_diff_eq!(pls.y_weights, expected_y_weights, epsilon = 1e-6);
        assert_abs_diff_eq!(pls.y_loadings, expected_y_loadings, epsilon = 1e-6);
    }

    #[test]
    fn test_sanity_check_pls_regression_constant_column_y() {
        let mut dataset = linnerud();
        let nrows = dataset.targets.nrows();
        dataset.targets.column_mut(0).assign(&Array1::ones(nrows));
        let pls = Pls::regression(3)
            .fit(&dataset)
            .expect("PLS fitting failed");

        // The results were checked against scikit-learn 0.24 PlsRegression
        let expected_x_weights = array![
            [0.6273573, 0.007081799, 0.7786994],
            [0.7493417, -0.277612681, -0.6011807],
            [0.2119194, 0.960666981, -0.1794690]
        ];

        let expected_x_loadings = array![
            [0.6273512, -0.22464538, 0.7786994],
            [0.6643156, -0.09871193, -0.6011807],
            [0.5125877, 1.01407380, -0.1794690]
        ];

        let expected_y_loadings = array![
            [0.0000000, 0.0000000, 0.0000000],
            [-0.4357300, 0.5828479, 0.2174802],
            [0.1353739, -0.2486423, -0.1810386]
        ];
        assert_abs_diff_eq!(pls.x_weights, expected_x_weights, epsilon = 1e-6);
        assert_abs_diff_eq!(pls.x_loadings, expected_x_loadings, epsilon = 1e-6);
        // For the PLSRegression with default parameters, y_loadings == y_weights
        assert_abs_diff_eq!(pls.y_loadings, expected_y_loadings, epsilon = 1e-6);
        assert_abs_diff_eq!(pls.y_weights, expected_y_loadings, epsilon = 1e-6);
    }

    #[test]
    fn test_sanity_check_pls_canonical() -> Result<()> {
        // Sanity check for PLSCanonical
        // The results were checked against the R-package plspm
        let dataset = linnerud();
        let pls = Pls::canonical(dataset.records().ncols()).fit(&dataset)?;

        let expected_x_weights = array![
            [-0.61330704, 0.25616119, -0.74715187],
            [-0.74697144, 0.11930791, 0.65406368],
            [-0.25668686, -0.95924297, -0.11817271]
        ];

        let expected_x_rotations = array![
            [-0.61330704, 0.41591889, -0.62297525],
            [-0.74697144, 0.31388326, 0.77368233],
            [-0.25668686, -0.89237972, -0.24121788]
        ];

        let expected_y_weights = array![
            [0.58989127, 0.7890047, 0.1717553],
            [0.77134053, -0.61351791, 0.16920272],
            [-0.23887670, -0.03267062, 0.97050016]
        ];

        let expected_y_rotations = array![
            [0.58989127, 0.7168115, 0.30665872],
            [0.77134053, -0.70791757, 0.19786539],
            [-0.23887670, -0.00343595, 0.94162826]
        ];

        let (x_weights, y_weights) = pls.weights();
        let (x_rotations, y_rotations) = pls.rotations();
        assert_abs_diff_eq!(
            expected_x_rotations.mapv(|v: f64| v.abs()),
            x_rotations.mapv(|v| v.abs()),
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(
            expected_x_weights.mapv(|v: f64| v.abs()),
            x_weights.mapv(|v| v.abs()),
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(
            expected_y_rotations.mapv(|v: f64| v.abs()),
            y_rotations.mapv(|v| v.abs()),
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(
            expected_y_weights.mapv(|v: f64| v.abs()),
            y_weights.mapv(|v| v.abs()),
            epsilon = 1e-7
        );

        let x_rotations_sign_flip = (x_rotations / &expected_x_rotations).mapv(|v| v.signum());
        let x_weights_sign_flip = (x_weights / &expected_x_weights).mapv(|v| v.signum());
        let y_rotations_sign_flip = (y_rotations / &expected_y_rotations).mapv(|v| v.signum());
        let y_weights_sign_flip = (y_weights / &expected_y_weights).mapv(|v| v.signum());
        assert_abs_diff_eq!(x_rotations_sign_flip, x_weights_sign_flip);
        assert_abs_diff_eq!(y_rotations_sign_flip, y_weights_sign_flip);

        assert_matrix_orthonormal(x_weights);
        assert_matrix_orthonormal(y_weights);

        let (x_scores, y_scores) = pls.scores();
        assert_matrix_orthogonal(x_scores);
        assert_matrix_orthogonal(y_scores);
        Ok(())
    }

    #[test]
    fn test_sanity_check_pls_canonical_random() {
        // Sanity check for PLSCanonical on random data
        // The results were checked against the R-package plspm
        let n = 500;
        let p_noise = 10;
        let q_noise = 5;

        // 2 latents vars:
        let mut rng = Xoshiro256Plus::seed_from_u64(100);
        let l1: Array1<f64> = Array1::random_using(n, StandardNormal, &mut rng);
        let l2: Array1<f64> = Array1::random_using(n, StandardNormal, &mut rng);
        let mut latents = Array::zeros((4, n));
        latents.row_mut(0).assign(&l1);
        latents.row_mut(0).assign(&l1);
        latents.row_mut(0).assign(&l2);
        latents.row_mut(0).assign(&l2);
        latents = latents.reversed_axes();

        let mut x = &latents + &Array2::<f64>::random_using((n, 4), StandardNormal, &mut rng);
        let mut y = latents + &Array2::<f64>::random_using((n, 4), StandardNormal, &mut rng);

        x = concatenate(
            Axis(1),
            &[
                x.view(),
                Array2::random_using((n, p_noise), StandardNormal, &mut rng).view(),
            ],
        )
        .unwrap();
        y = concatenate(
            Axis(1),
            &[
                y.view(),
                Array2::random_using((n, q_noise), StandardNormal, &mut rng).view(),
            ],
        )
        .unwrap();

        let ds = Dataset::new(x, y);
        let pls = Pls::canonical(3)
            .fit(&ds)
            .expect("PLS canonical fitting failed");

        let (x_weights, y_weights) = pls.weights();
        assert_matrix_orthonormal(x_weights);
        assert_matrix_orthonormal(y_weights);

        let (x_scores, y_scores) = pls.scores();
        assert_matrix_orthogonal(x_scores);
        assert_matrix_orthogonal(y_scores);
    }

    #[test]
    fn test_scale_and_stability() -> Result<()> {
        // scale=True is equivalent to scale=False on centered/scaled data
        // This allows to check numerical stability over platforms as well

        let ds = linnerud();
        let (x_s, y_s, ..) = utils::center_scale_dataset(&ds, true);
        let ds_s = Dataset::new(x_s, y_s);

        let ds_score = Pls::regression(2)
            .scale(true)
            .tolerance(1e-3)
            .fit(&ds)?
            .transform(ds.to_owned());
        let ds_s_score = Pls::regression(2)
            .scale(false)
            .tolerance(1e-3)
            .fit(&ds_s)?
            .transform(ds_s.to_owned());

        assert_abs_diff_eq!(ds_s_score.records(), ds_score.records(), epsilon = 1e-4);
        assert_abs_diff_eq!(ds_s_score.targets(), ds_score.targets(), epsilon = 1e-4);
        Ok(())
    }

    #[test]
    fn test_one_component_equivalence() -> Result<()> {
        // PlsRegression, PlsSvd and PLSCanonical should all be equivalent when n_components is 1
        let ds = linnerud();
        let ds2 = linnerud();
        let regression = Pls::regression(1).fit(&ds)?.transform(ds);
        let canonical = Pls::canonical(1).fit(&ds2)?.transform(ds2);

        assert_abs_diff_eq!(regression.records(), canonical.records(), epsilon = 1e-7);
        Ok(())
    }

    #[test]
    fn test_convergence_fail() {
        let ds = linnerud();
        assert!(
            Pls::canonical(ds.records().nfeatures())
                .max_iterations(2)
                .fit(&ds)
                .is_err(),
            "PLS power method should not converge, hence raise an error"
        );
    }

    #[test]
    fn test_bad_component_number() {
        let ds = linnerud();
        assert!(
            Pls::cca(ds.records().nfeatures() + 1).fit(&ds).is_err(),
            "n_components too large should raise an error"
        );
        assert!(
            Pls::canonical(0).fit(&ds).is_err(),
            "n_components=0 should raise an error"
        );
    }

    #[test]
    fn test_singular_value_helpers() -> Result<()> {
        // Make sure SVD and power method give approximately the same results
        let ds = linnerud();

        let (mut u1, mut v1, _) = PlsParams::new(2)
            .check()?
            .get_first_singular_vectors_power_method(ds.records(), ds.targets(), true)?;
        let (mut u2, mut v2) = PlsParams::new(2)
            .check()?
            .get_first_singular_vectors_svd(ds.records(), ds.targets())?;

        utils::svd_flip_1d(&mut u1, &mut v1);
        utils::svd_flip_1d(&mut u2, &mut v2);

        let rtol = 1e-1;
        assert_abs_diff_eq!(u1, u2, epsilon = rtol);
        assert_abs_diff_eq!(v1, v2, epsilon = rtol);
        Ok(())
    }

    macro_rules! test_pls_algo_nipals_svd {
        ($($name:ident, )*) => {
            paste::item! {
                $(
                    #[test]
                    fn [<test_pls_$name>]() -> Result<()> {
                        let ds = linnerud();
                        let pls = Pls::[<$name>](3).fit(&ds)?;
                        let ds1 = pls.transform(ds.to_owned());
                        let ds2 = Pls::[<$name>](3).algorithm(Algorithm::Svd).fit(&ds)?.transform(ds);
                        assert_abs_diff_eq!(ds1.records(), ds2.records(), epsilon=1e-2);
                        let exercices = array![[14., 146., 61.], [6., 80., 60.]];
                        let physios = pls.predict(exercices);
                        println!("Physiologicals = {:?}", physios.targets());
                        Ok(())
                    }
                )*
            }
        };
    }

    test_pls_algo_nipals_svd! {
        canonical, regression,
    }

    #[test]
    fn test_cca() -> Result<()> {
        // values checked against scikit-learn 0.24.1 CCA
        let ds = linnerud();
        let cca = Pls::cca(3).fit(&ds)?;
        let ds = cca.transform(ds);
        let expected_x = array![
            [0.09597886, 0.13862931, -1.0311966],
            [-0.7170194, 0.25195026, -0.83049671],
            [-0.76492193, 0.37601463, 1.20714686],
            [-0.03734329, -0.9746487, 0.79363542],
            [0.42809962, -0.50053551, 0.40089685],
            [-0.54141144, -0.29403268, -0.47221389],
            [-0.29901672, -0.67023009, 0.17945745],
            [-0.11425233, -0.43360723, -0.47235823],
            [1.29212153, -0.9373391, 0.02572464],
            [-0.17770025, 3.4785377, 0.8486413],
            [0.39344638, -1.28718499, 1.43816035],
            [0.52667844, 0.82080301, -0.02624471],
            [0.74616393, 0.54578854, 0.01825073],
            [-1.42623443, -0.00884605, -0.24019883],
            [-0.72026991, -0.73588273, 0.2241694],
            [0.4237932, 0.99977428, -0.1667137],
            [-0.88437821, -0.73784626, -0.01073894],
            [1.05159992, 0.26381077, -0.83138216],
            [1.26196754, -0.18618728, -0.12863494],
            [-0.53730151, -0.10896789, -0.92590428]
        ];
        assert_abs_diff_eq!(expected_x, ds.records(), epsilon = 1e-2);
        Ok(())
    }

    #[test]
    fn test_transform_and_inverse() -> Result<()> {
        let ds = linnerud();
        let pls = Pls::canonical(3).fit(&ds)?;

        let ds_proj = pls.transform(ds);
        let ds_orig = pls.inverse_transform(ds_proj);

        let ds = linnerud();
        assert_abs_diff_eq!(ds.records(), ds_orig.records(), epsilon = 1e-6);
        assert_abs_diff_eq!(ds.targets(), ds_orig.targets(), epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_pls_constant_y() {
        // Checks constant residual error when y is constant.
        let n = 100;
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let x = Array2::<f64>::random_using((n, 3), StandardNormal, &mut rng);
        let y = Array2::zeros((n, 1));
        let ds = Dataset::new(x, y);
        assert!(matches!(
            Pls::regression(2).fit(&ds).unwrap_err(),
            PlsError::PowerMethodConstantResidualError()
        ));
    }
}
