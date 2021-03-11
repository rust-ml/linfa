use crate::errors::{PlsError, Result};
use linfa::{traits::Fit, DatasetBase, Float};
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_linalg::{svd::*, Lapack, Scalar};
use ndarray_stats::QuantileExt;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct Pls<F: Float> {
    n_components: usize,
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
    coeffs: Array2<F>,
    n_iters: Array1<usize>,
}

impl<F: Float> Pls<F> {
    pub fn regression(n_components: usize) -> PlsParams<F> {
        PlsParams::new(n_components)
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

#[derive(Debug, Clone)]
pub struct PlsParams<F: Float> {
    n_components: usize,
    max_iter: usize,
    tolerance: F,
}

impl<F: Float> PlsParams<F> {
    pub fn new(n_components: usize) -> PlsParams<F> {
        PlsParams {
            n_components,
            max_iter: 500,
            tolerance: F::from(1e-6).unwrap(),
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
}

impl<'a, F: Float + Scalar + Lapack> Fit<'a, Array2<F>, Array2<F>> for PlsParams<F> {
    type Object = Result<Pls<F>>;

    fn fit(&self, dataset: &DatasetBase<Array2<F>, Array2<F>>) -> Result<Pls<F>> {
        let x = dataset.records();
        let y = dataset.targets();

        let n = x.nrows();
        let p = x.ncols();
        let q = y.ncols();

        let n_components = self.n_components;

        // With PLSRegression n_components is bounded by the rank of (x.T x)
        // see Wegelin page 25
        let rank_upper_bound = p;
        if 1 > n_components || n_components > rank_upper_bound {
            return Err(PlsError::new(format!(
                "n_components should be in [1, {}], got {}",
                rank_upper_bound, n_components
            )));
        }

        // Scale (in place)
        let (mut xk, mut yk, x_mean, y_mean, x_std, y_std) = center_scale(&x, &y);

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

            // nipals algorithm
            // Replace columns that are all close to zero with zeros
            for mut yj in yk.gencolumns_mut() {
                if *(yj.mapv(|y| y.abs()).max().unwrap()) < F::from(10.).unwrap() * eps {
                    yj.assign(&Array1::zeros(yj.len()));
                }
            }

            let (mut _x_weights, mut _y_weights, n_iter) =
                self.get_first_singular_vectors_power_method(&xk, &yk);
            n_iters[i] = n_iter;

            // svd_flip(x_weights, y_weights)
            let biggest_abs_val_idx = _x_weights.mapv(|v| v.abs()).argmax().unwrap();
            let sign: F = _x_weights[biggest_abs_val_idx].signum();
            _x_weights.map_inplace(|v| *v *= sign);
            _y_weights.map_inplace(|v| *v *= sign);

            // compute scores, i.e. the projections of x and Y
            let _x_scores = xk.dot(&_x_weights);
            let _y_scores = yk.dot(&_y_weights) / _y_weights.dot(&_y_weights);

            // Deflation: subtract rank-one approx to obtain xk+1 and yk+1
            let _x_loadings = _x_scores.dot(&xk) / _x_scores.dot(&_x_scores);
            xk = xk - outer(&_x_scores, &_x_loadings); // outer product

            // regress yk on x_score
            let _y_loadings = _x_scores.dot(&yk) / _x_scores.dot(&_x_scores);
            yk = yk - outer(&_x_scores, &_y_loadings); // outer product

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
        let x_rotations = x_weights.dot(&pinv2(&x_loadings.t().dot(&x_weights)));
        let y_rotations = y_weights.dot(&pinv2(&y_loadings.t().dot(&y_weights)));

        let mut coeffs = x_rotations.dot(&y_loadings.t());
        coeffs = &coeffs * &y_std;

        Ok(Pls {
            n_components,
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
            coeffs,
            n_iters,
        })
    }
}

impl<F: Float> PlsParams<F> {
    /// Return the first left and right singular vectors of x'Y.
    /// Provides an alternative to the svd(x'Y) and uses the power method instead.
    fn get_first_singular_vectors_power_method(
        &self,
        x: &ArrayBase<impl Data<Elem = F>, Ix2>,
        y: &ArrayBase<impl Data<Elem = F>, Ix2>,
    ) -> (Array1<F>, Array1<F>, usize) {
        let eps = F::epsilon();

        let mut y_score = Array1::ones(y.ncols());
        for col in y.t().genrows() {
            if *col.mapv(|v| v.abs()).max().unwrap() > eps {
                y_score = col.to_owned();
                break;
            }
        }

        // init to big value for first convergence check
        let mut x_weights_old = Array1::<F>::from_elem(x.ncols(), F::from(100.).unwrap());

        let mut n_iter = 1;
        let mut x_weights = Array1::<F>::ones(x.ncols());
        let mut y_weights = Array1::<F>::ones(y.ncols());
        while n_iter < self.max_iter {
            x_weights = x.t().dot(&y_score) / y_score.dot(&y_score);
            x_weights /= (x_weights.dot(&x_weights)).sqrt() + eps;
            let x_score = x.dot(&x_weights);

            y_weights = y.t().dot(&x_score) / x_score.dot(&x_score);
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
                "Singular vector computation power method: max iterations ({}) reached",
                self.max_iter
            );
        }

        (x_weights, y_weights, n_iter)
    }
}

fn center_scale<F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    y: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> (
    Array2<F>,
    Array2<F>,
    Array1<F>,
    Array1<F>,
    Array1<F>,
    Array1<F>,
) {
    let (xnorm, x_mean, x_std) = normalize(&x);
    let (ynorm, y_mean, y_std) = normalize(&y);
    (xnorm, ynorm, x_mean, y_mean, x_std, y_std)
}

fn outer<F: Float>(
    a: &ArrayBase<impl Data<Elem = F>, Ix1>,
    b: &ArrayBase<impl Data<Elem = F>, Ix1>,
) -> Array2<F> {
    let mut outer = Array2::zeros((a.len(), b.len()));
    Zip::from(outer.genrows_mut()).and(a).apply(|mut out, ai| {
        out.assign(&b.mapv(|v| *ai * v));
    });
    outer
}

fn pinv2<F: Float + Scalar + Lapack>(x: &ArrayBase<impl Data<Elem = F>, Ix2>) -> Array2<F> {
    let (opt_u, s, opt_vh) = x.svd(true, true).unwrap();
    let u = opt_u.unwrap();
    let vh = opt_vh.unwrap();
    let cond = F::from(*s.max().unwrap()).unwrap()
        * F::from(x.nrows().max(x.ncols())).unwrap()
        * F::epsilon();

    let rank = s.fold(0, |mut acc, v| {
        if F::from(*v).unwrap() > cond {
            acc += 1
        };
        acc
    });

    let mut ucut = u.slice(s![.., ..rank]).to_owned();
    ucut /= &s.slice(s![..rank]).mapv(|v| F::from(v).unwrap());
    ucut.dot(&vh.slice(s![..rank, ..]))
        .mapv(|v| v.conj())
        .t()
        .to_owned()
}

pub fn normalize<F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix2>,
) -> (Array2<F>, Array1<F>, Array1<F>) {
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let mut x_std = x.std_axis(Axis(0), F::one());
    x_std.mapv_inplace(|v| if v == F::zero() { F::one() } else { v });
    let xnorm = (x - &x_mean) / &x_std;

    (xnorm, x_mean, x_std)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa_datasets::linnerud;
    use ndarray::array;

    #[test]
    fn test_outer() {
        let a = array![1., 2., 3.];
        let b = array![2., 3.];
        let expected = array![[2., 3.], [4., 6.], [6., 9.]];
        assert_abs_diff_eq!(expected, outer(&a, &b));
    }

    #[test]
    fn test_pinv2() {
        let a = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 10.]];
        let a_pinv2 = pinv2(&a);
        assert_abs_diff_eq!(a.dot(&a_pinv2), Array2::eye(3), epsilon = 1e-6)
    }

    #[test]
    fn test_pls_fit() {
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
    fn test_pls_fit_constant_column_y() {
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
}
