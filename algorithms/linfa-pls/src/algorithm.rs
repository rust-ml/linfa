use crate::errors::{PlsError, Result};
use crate::utils;
use linfa::{traits::Fit, traits::Transformer, Dataset, DatasetBase, Float};
use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{svd::*, Lapack, Scalar};
use ndarray_stats::QuantileExt;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(PartialEq, Debug, Clone, Copy)]
enum Algorithm {
    Nipals,
    Svd,
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct Pls<F: Float> {
    x_mean: Array1<F>,
    x_std: Array1<F>,
    y_mean: Array1<F>,
    y_std: Array1<F>,
    norm_y_weights: bool,
    x_weights: Array2<F>,  // U
    y_weights: Array2<F>,  // V
    x_scores: Array2<F>,   // xi
    y_scores: Array2<F>,   // Omega
    x_loadings: Array2<F>, // Gamma
    y_loadings: Array2<F>, // Delta
    x_rotations: Array2<F>,
    y_rotations: Array2<F>,
    coeffs: Array2<F>,
    n_iters: Array1<usize>,
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum DeflationMode {
    Regression,
    Canonical,
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Mode {
    A,
    B,
}

impl<F: Float> Pls<F> {
    pub fn regression(n_components: usize) -> PlsParams<F> {
        PlsParams::new(n_components)
    }

    pub fn canonical(n_components: usize) -> PlsParams<F> {
        PlsParams::new(n_components)
            .deflation_mode(DeflationMode::Canonical)
            .mode(Mode::A)
    }

    pub fn cca(n_components: usize) -> PlsParams<F> {
        PlsParams::new(n_components)
            .deflation_mode(DeflationMode::Canonical)
            .mode(Mode::B)
    }

    pub fn means(&self) -> (&Array1<F>, &Array1<F>) {
        (&self.x_mean, &self.y_mean)
    }

    pub fn stds(&self) -> (&Array1<F>, &Array1<F>) {
        (&self.x_std, &self.y_std)
    }

    pub fn weights(&self) -> (&Array2<F>, &Array2<F>) {
        (&self.x_weights, &self.y_weights)
    }

    pub fn scores(&self) -> (&Array2<F>, &Array2<F>) {
        (&self.x_scores, &self.y_scores)
    }

    pub fn loadings(&self) -> (&Array2<F>, &Array2<F>) {
        (&self.x_loadings, &self.y_loadings)
    }

    pub fn rotations(&self) -> (&Array2<F>, &Array2<F>) {
        (&self.x_rotations, &self.y_rotations)
    }

    pub fn coeffs(&self) -> &Array2<F> {
        &self.coeffs
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
        let mut x = dataset.records() - &self.x_mean;
        x /= &self.x_std;
        let mut y = dataset.targets() - &self.y_mean;
        y /= &self.y_std;
        // Apply rotation
        Dataset::new(x.dot(&self.x_rotations), y.dot(&self.y_rotations))
    }
}

#[derive(Debug, Clone)]
pub struct PlsParams<F: Float> {
    n_components: usize,
    max_iter: usize,
    tolerance: F,
    scale: bool,
    algorithm: Algorithm,
    deflation_mode: DeflationMode,
    mode: Mode,
}

impl<F: Float> PlsParams<F> {
    pub fn new(n_components: usize) -> PlsParams<F> {
        PlsParams {
            n_components,
            max_iter: 500,
            tolerance: F::from(1e-6).unwrap(),
            scale: true,
            algorithm: Algorithm::Nipals,
            deflation_mode: DeflationMode::Regression,
            mode: Mode::A,
        }
    }

    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    fn deflation_mode(mut self, deflation_mode: DeflationMode) -> Self {
        self.deflation_mode = deflation_mode;
        self
    }

    fn mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }
}

impl<F: Float + Scalar + Lapack, D: Data<Elem = F>> Fit<'_, ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>
    for PlsParams<F>
{
    type Object = Result<Pls<F>>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>) -> Result<Pls<F>> {
        let x = dataset.records();
        let y = dataset.targets();

        let n = x.nrows();
        let p = x.ncols();
        let q = y.ncols();

        let n_components = self.n_components;
        let rank_upper_bound = match self.deflation_mode {
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
            return Err(PlsError::new(format!(
                "n_components should be in [1, {}], got {}",
                rank_upper_bound, n_components
            )));
        }
        let norm_y_weights = self.deflation_mode == DeflationMode::Canonical;
        // Scale (in place)
        let (mut xk, mut yk, x_mean, y_mean, x_std, y_std) =
            utils::center_scale_xy(&x, &y, self.scale);

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
        for (i, k) in (0..n_components).enumerate() {
            // Find first left and right singular vectors of the x.T.dot(Y)
            // cross-covariance matrix.

            let (mut _x_weights, mut _y_weights) = match self.algorithm {
                Algorithm::Nipals => {
                    // Replace columns that are all close to zero with zeros
                    for mut yj in yk.gencolumns_mut() {
                        if *(yj.mapv(|y| y.abs()).max().unwrap()) < F::from(10.).unwrap() * eps {
                            yj.assign(&Array1::zeros(yj.len()));
                        }
                    }

                    let (mut _x_weights, mut _y_weights, n_iter) =
                        self.get_first_singular_vectors_power_method(&xk, &yk, norm_y_weights);
                    n_iters[i] = n_iter;
                    (_x_weights, _y_weights)
                }
                Algorithm::Svd => self.get_first_singular_vectors_svd(&xk, &yk),
            };
            // svd_flip_1d(x_weights, y_weights)
            let biggest_abs_val_idx = _x_weights.mapv(|v| v.abs()).argmax().unwrap();
            let sign: F = _x_weights[biggest_abs_val_idx].signum();
            _x_weights.map_inplace(|v| *v *= sign);
            _y_weights.map_inplace(|v| *v *= sign);

            // compute scores, i.e. the projections of x and Y
            let _x_scores = xk.dot(&_x_weights);
            let y_ss = if norm_y_weights {
                F::from(1.).unwrap()
            } else {
                _y_weights.dot(&_y_weights)
            };
            let _y_scores = yk.dot(&_y_weights) / y_ss;

            // Deflation: subtract rank-one approx to obtain xk+1 and yk+1
            let _x_loadings = _x_scores.dot(&xk) / _x_scores.dot(&_x_scores);
            xk = xk - utils::outer(&_x_scores, &_x_loadings); // outer product

            let _y_loadings = match self.deflation_mode {
                DeflationMode::Canonical => {
                    // regress yk on y_score
                    let _y_loadings = _y_scores.dot(&yk) / _y_scores.dot(&_y_scores);
                    yk = yk - utils::outer(&_y_scores, &_y_loadings); // outer product
                    _y_loadings
                }
                DeflationMode::Regression => {
                    // regress yk on x_score
                    let _y_loadings = _x_scores.dot(&yk) / _x_scores.dot(&_x_scores);
                    yk = yk - utils::outer(&_x_scores, &_y_loadings); // outer product
                    _y_loadings
                }
            };

            x_weights.column_mut(k).assign(&_x_weights);
            y_weights.column_mut(k).assign(&_y_weights);
            x_scores.column_mut(k).assign(&_x_scores);
            y_scores.column_mut(k).assign(&_y_scores);
            x_loadings.column_mut(k).assign(&_x_loadings);
            y_loadings.column_mut(k).assign(&_y_loadings);
        }
        // x was approximated as xi . Gamma.T + x_(R+1) xi . Gamma.T is a sum
        // of n_components rank-1 matrices. x_(R+1) is whatever is left
        // to fully reconstruct x, and can be 0 if x is of rank n_components.
        // Similiarly, Y was approximated as Omega . Delta.T + Y_(R+1)

        // Compute transformation matrices (rotations_). See User Guide.
        let x_rotations = x_weights.dot(&utils::pinv2(&x_loadings.t().dot(&x_weights), None));
        let y_rotations = y_weights.dot(&utils::pinv2(&y_loadings.t().dot(&y_weights), None));

        let mut coeffs = x_rotations.dot(&y_loadings.t());
        coeffs = &coeffs * &y_std;

        Ok(Pls {
            x_mean,
            x_std,
            y_mean,
            y_std,
            norm_y_weights,
            x_weights,
            y_weights,
            x_scores,
            y_scores,
            x_loadings,
            y_loadings,
            x_rotations,
            y_rotations,
            coeffs,
            n_iters,
        })
    }
}

impl<F: Float + Scalar + Lapack> PlsParams<F> {
    /// Return the first left and right singular vectors of x'Y.
    /// Provides an alternative to the svd(x'Y) and uses the power method instead.
    fn get_first_singular_vectors_power_method(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        y: &ArrayBase<impl Data<Elem = F>, Ix2>,
        norm_y_weights: bool,
    ) -> (Array1<F>, Array1<F>, usize) {
        let eps = F::epsilon();

        let mut y_score = Array1::ones(y.ncols());
        for col in y.t().genrows() {
            if *col.mapv(|v| v.abs()).max().unwrap() > eps {
                y_score = col.to_owned();
                break;
            }
        }

        let mut x_pinv = None;
        let mut y_pinv = None;
        if self.mode == Mode::B {
            x_pinv = Some(utils::pinv2(&x, Some(F::from(10.).unwrap() * eps)));
            y_pinv = Some(utils::pinv2(&y, Some(F::from(10.).unwrap() * eps)));
        }

        // init to big value for first convergence check
        let mut x_weights_old = Array1::<F>::from_elem(x.ncols(), F::from(100.).unwrap());

        let mut n_iter = 1;
        let mut x_weights = Array1::<F>::ones(x.ncols());
        let mut y_weights = Array1::<F>::ones(y.ncols());
        while n_iter < self.max_iter {
            x_weights = match self.mode {
                Mode::A => x.t().dot(&y_score) / y_score.dot(&y_score),
                Mode::B => x_pinv.to_owned().unwrap().dot(&y_score),
            };
            x_weights /= (x_weights.dot(&x_weights)).sqrt() + eps;
            let x_score = x.dot(&x_weights);

            y_weights = match self.mode {
                Mode::A => y.t().dot(&x_score) / x_score.dot(&x_score),
                Mode::B => y_pinv.to_owned().unwrap().dot(&y_score),
            };

            if norm_y_weights {
                y_weights /= (y_weights.dot(&y_weights)).sqrt() + eps
            }

            let ya = y.dot(&y_weights);
            let yb = y_weights.dot(&y_weights) + eps;
            y_score = ya.mapv(|v| v / yb);

            let x_weights_diff = &x_weights - &x_weights_old;
            if x_weights_diff.dot(&x_weights_diff) < self.tolerance || y.ncols() == 1 {
                break;
            } else {
                x_weights_old = x_weights.to_owned();
                n_iter += 1;
            }
        }
        if n_iter == self.max_iter {
            println!(
                "Warning: Singular vector computation power method: max iterations ({}) reached",
                self.max_iter
            );
        }

        (x_weights, y_weights, n_iter)
    }
}

impl<F: Float + Scalar + Lapack> PlsParams<F> {
    fn get_first_singular_vectors_svd(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        y: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array1<F>, Array1<F>) {
        let c = x.dot(y);
        let (u, _, vt) = c.svd(true, true).unwrap();
        let u = u.unwrap().row(0).to_owned();
        let vt = vt.unwrap().row(0).to_owned();
        (u, vt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa_datasets::linnerud;
    use ndarray::{array, Array};

    fn assert_matrix_orthogonal(m: &Array2<f64>) {
        assert_abs_diff_eq!(&m.t().dot(m), &Array::eye(m.ncols()), epsilon = 1e-7);
    }

    fn assert_matrix_diagonal(m: &Array2<f64>) {
        let k = m.t().dot(m);
        assert_abs_diff_eq!(&k, &(&Array::eye(m.ncols()) * &k.diag()), epsilon = 1e-7);
    }

    #[test]
    fn test_pls_canonical_basics() -> Result<()> {
        // Basic checks for PLSCanonical
        let dataset = linnerud();
        let x = dataset.records();
        let y = dataset.targets();

        let pls = Pls::canonical(x.ncols()).fit(&dataset)?;

        let (x_weights, y_weights) = pls.weights();
        assert_matrix_orthogonal(x_weights);
        assert_matrix_orthogonal(y_weights);

        let (x_scores, y_scores) = pls.scores();
        assert_matrix_diagonal(x_scores);
        assert_matrix_diagonal(y_scores);

        // Check X = TP' and Y = UQ'
        let (p, q) = pls.loadings();
        let t = x_scores;
        let u = y_scores;

        // Need to scale first
        let (xc, yc, ..) = utils::center_scale_xy(&x, &y, true);
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
        let mut dataset = linnerud();
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
            expected_x_rotations.mapv(|v| v.abs()),
            x_rotations.mapv(|v| v.abs()),
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(
            expected_x_weights.mapv(|v| v.abs()),
            x_weights.mapv(|v| v.abs()),
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(
            expected_y_rotations.mapv(|v| v.abs()),
            y_rotations.mapv(|v| v.abs()),
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(
            expected_y_weights.mapv(|v| v.abs()),
            y_weights.mapv(|v| v.abs()),
            epsilon = 1e-7
        );

        let x_rotations_sign_flip = (x_rotations / &expected_x_rotations).mapv(|v| v.signum());
        let x_weights_sign_flip = (x_weights / &expected_x_weights).mapv(|v| v.signum());
        let y_rotations_sign_flip = (y_rotations / &expected_y_rotations).mapv(|v| v.signum());
        let y_weights_sign_flip = (y_weights / &expected_y_weights).mapv(|v| v.signum());
        assert_abs_diff_eq!(x_rotations_sign_flip, x_weights_sign_flip);
        assert_abs_diff_eq!(y_rotations_sign_flip, y_weights_sign_flip);

        assert_matrix_orthogonal(x_weights);
        assert_matrix_orthogonal(y_weights);

        let (x_scores, y_scores) = pls.scores();
        assert_matrix_diagonal(x_scores);
        assert_matrix_diagonal(y_scores);
        Ok(())
    }
}
