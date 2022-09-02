use approx::{abs_diff_eq, abs_diff_ne};
use linfa_linalg::norm::Norm;
#[cfg(not(feature = "blas"))]
use linfa_linalg::qr::QRInto;
use ndarray::linalg::general_mat_mul;
use ndarray::{
    s, Array, Array1, Array2, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, CowArray, Data,
    Dimension, Ix2, RemoveAxis,
};
#[cfg(feature = "blas")]
use ndarray_linalg::InverseHInto;

use linfa::dataset::{WithLapack, WithoutLapack};
use linfa::traits::{Fit, PredictInplace};
use linfa::{
    dataset::{AsMultiTargets, AsSingleTargets, AsTargets, Records},
    DatasetBase, Float,
};

use super::{
    hyperparams::{ElasticNetValidParams, MultiTaskElasticNetValidParams},
    ElasticNet, ElasticNetError, MultiTaskElasticNet, Result,
};

impl<F, D, T> Fit<ArrayBase<D, Ix2>, T, ElasticNetError> for ElasticNetValidParams<F>
where
    F: Float,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = F>,
{
    type Object = ElasticNet<F>;

    /// Fit an elastic net model given a feature matrix `x` and a target
    /// variable `y`.
    ///
    /// The feature matrix `x` must have shape `(n_samples, n_features)`
    ///
    /// The target variable `y` must have shape `(n_samples)`
    ///
    /// Returns a `FittedElasticNet` object which contains the fitted
    /// parameters and can be used to `predict` values of the target variable
    /// for new feature values.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let target = dataset.as_single_targets();

        let (intercept, y) = compute_intercept(self.with_intercept(), target);
        let (hyperplane, duality_gap, n_steps) = coordinate_descent(
            dataset.records().view(),
            y.view(),
            self.tolerance(),
            self.max_iterations(),
            self.l1_ratio(),
            self.penalty(),
        );
        let intercept = intercept.into_scalar();

        let y_est = dataset.records().dot(&hyperplane) + intercept;

        // try to calculate the variance
        let variance = variance_params(dataset, y_est.view());

        Ok(ElasticNet {
            hyperplane,
            intercept,
            duality_gap,
            n_steps,
            variance,
        })
    }
}

impl<F, D, T> Fit<ArrayBase<D, Ix2>, T, ElasticNetError> for MultiTaskElasticNetValidParams<F>
where
    F: Float,
    T: AsMultiTargets<Elem = F>,
    D: Data<Elem = F>,
{
    type Object = MultiTaskElasticNet<F>;

    /// Fit a multi-task Elastic Net model given a feature matrix `x` and a target
    /// matrix `y`.
    ///
    /// The feature matrix `x` must have shape `(n_samples, n_features)`
    ///
    /// The target variable `y` must have shape `(n_samples, n_tasks)`
    ///
    /// Returns a `FittedMultiTaskElasticNet` object which contains the fitted
    /// parameters and can be used to `predict` values of the target variables
    /// for new feature values.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let targets = dataset.targets().as_multi_targets();
        let (intercept, y) = compute_intercept(self.with_intercept(), targets);

        let (hyperplane, duality_gap, n_steps) = block_coordinate_descent(
            dataset.records().view(),
            y.view(),
            self.tolerance(),
            self.max_iterations(),
            self.l1_ratio(),
            self.penalty(),
        );

        let y_est = dataset.records().dot(&hyperplane) + &intercept;

        // try to calculate the variance
        let variance = variance_params(dataset, y_est.view());

        Ok(MultiTaskElasticNet {
            hyperplane,
            intercept,
            duality_gap,
            n_steps,
            variance,
        })
    }
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<F>> for ElasticNet<F> {
    /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
    /// `predict` returns the target variable according to elastic net
    /// learned from the training data distribution.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<F>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        *y = x.dot(&self.hyperplane) + self.intercept;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        Array1::zeros(x.nrows())
    }
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array2<F>>
    for MultiTaskElasticNet<F>
{
    /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
    /// `predict` returns the target variable according to elastic net
    /// learned from the training data distribution.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array2<F>) {
        assert_eq!(
            x.nrows(),
            y.nrows(),
            "The number of data points must match the number of output targets."
        );

        *y = x.dot(&self.hyperplane) + &self.intercept;
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
        // TODO: fix, should be (x.nrows(), y.ncols())
        Array2::zeros((x.nrows(), x.nrows()))
    }
}

/// View the fitted parameters and make predictions with a fitted
/// elastic net model
impl<F: Float> ElasticNet<F> {
    /// Get the fitted hyperplane
    pub fn hyperplane(&self) -> &Array1<F> {
        &self.hyperplane
    }

    /// Get the fitted intercept, 0. if no intercept was fitted
    pub fn intercept(&self) -> F {
        self.intercept
    }

    /// Get the number of steps taken in optimization algorithm
    pub fn n_steps(&self) -> u32 {
        self.n_steps
    }

    /// Get the duality gap at the end of the optimization algorithm
    pub fn duality_gap(&self) -> F {
        self.duality_gap
    }

    /// Calculate the Z score
    pub fn z_score(&self) -> Result<Array1<F>> {
        self.variance
            .as_ref()
            .map(|variance| {
                self.hyperplane
                    .iter()
                    .zip(variance.iter())
                    .map(|(a, b)| *a / b.sqrt())
                    .collect()
            })
            .map_err(|err| err.clone())
    }

    /// Calculate the confidence level
    pub fn confidence_95th(&self) -> Result<Array1<(F, F)>> {
        // the 95th percentile of our confidence level
        let p = F::cast(1.645);

        self.variance
            .as_ref()
            .map(|variance| {
                self.hyperplane
                    .iter()
                    .zip(variance.iter())
                    .map(|(a, b)| (*a - p * b.sqrt(), *a + p * b.sqrt()))
                    .collect()
            })
            .map_err(|err| err.clone())
    }
}

/// View the fitted parameters and make predictions with a fitted
/// elastic net model
impl<F: Float> MultiTaskElasticNet<F> {
    /// Get the fitted hyperplane
    pub fn hyperplane(&self) -> &Array2<F> {
        &self.hyperplane
    }

    /// Get the fitted intercept, [0., ..., 0.] if no intercept was fitted
    /// Note that there are as many intercepts as tasks
    pub fn intercept(&self) -> &Array1<F> {
        &self.intercept
    }

    /// Get the number of steps taken in optimization algorithm
    pub fn n_steps(&self) -> u32 {
        self.n_steps
    }

    /// Get the duality gap at the end of the optimization algorithm
    pub fn duality_gap(&self) -> F {
        self.duality_gap
    }

    /// Calculate the Z score
    pub fn z_score(&self) -> Result<Array2<F>> {
        self.variance
            .as_ref()
            .map(|variance| {
                ndarray::Zip::from(&self.hyperplane)
                    .and_broadcast(variance)
                    .map_collect(|a, b| *a / b.sqrt())
            })
            .map_err(|err| err.clone())
    }

    /// Calculate the confidence level
    pub fn confidence_95th(&self) -> Result<Array2<(F, F)>> {
        // the 95th percentile of our confidence level
        let p = F::cast(1.645);

        self.variance
            .as_ref()
            .map(|variance| {
                ndarray::Zip::from(&self.hyperplane)
                    .and_broadcast(variance)
                    .map_collect(|a, b| (*a - p * b.sqrt(), *a + p * b.sqrt()))
            })
            .map_err(|err| err.clone())
    }
}

fn coordinate_descent<'a, F: Float>(
    x: ArrayView2<'a, F>,
    y: ArrayView1<'a, F>,
    tol: F,
    max_steps: u32,
    l1_ratio: F,
    penalty: F,
) -> (Array1<F>, F, u32) {
    let n_samples = F::cast(x.nrows());
    let n_features = x.ncols();
    // the parameters of the model
    let mut w = Array1::<F>::zeros(n_features);
    // the residuals: `y - X*w` (since w=0, this is just `y` for now),
    // the residuals are updated during the algorithm as the parameters change
    let mut r = y.to_owned();
    let mut n_steps = 0u32;
    let norm_cols_x = x.map_axis(Axis(0), |col| col.dot(&col));
    let mut gap = F::one() + tol;
    let d_w_tol = tol;
    let tol = tol * y.dot(&y);
    while n_steps < max_steps {
        let mut w_max = F::zero();
        let mut d_w_max = F::zero();
        for j in 0..n_features {
            if abs_diff_eq!(norm_cols_x[j], F::zero()) {
                continue;
            }
            let old_w_j = w[j];
            let x_j: ArrayView1<F> = x.slice(s![.., j]);
            if abs_diff_ne!(old_w_j, F::zero()) {
                r.scaled_add(old_w_j, &x_j);
            }
            let tmp: F = x_j.dot(&r);
            w[j] = tmp.signum() * F::max(tmp.abs() - n_samples * l1_ratio * penalty, F::zero())
                / (norm_cols_x[j] + n_samples * (F::one() - l1_ratio) * penalty);
            if abs_diff_ne!(w[j], F::zero()) {
                r.scaled_add(-w[j], &x_j);
            }
            let d_w_j = (w[j] - old_w_j).abs();
            d_w_max = F::max(d_w_max, d_w_j);
            w_max = F::max(w_max, w[j].abs());
        }
        n_steps += 1;

        if n_steps == max_steps - 1 || abs_diff_eq!(w_max, F::zero()) || d_w_max / w_max < d_w_tol {
            // We've hit one potential stopping criteria
            // check duality gap for ultimate stopping criterion
            gap = duality_gap(x.view(), y.view(), w.view(), r.view(), l1_ratio, penalty);
            if gap < tol {
                break;
            }
        }
    }
    (w, gap, n_steps)
}

fn block_coordinate_descent<'a, F: Float>(
    x: ArrayView2<'a, F>,
    y: ArrayView2<'a, F>,
    tol: F,
    max_steps: u32,
    l1_ratio: F,
    penalty: F,
) -> (Array2<F>, F, u32) {
    let n_samples = F::cast(x.nrows());
    let n_features = x.ncols();
    let n_tasks = y.ncols();
    // the parameters of the model
    let mut w = Array2::<F>::zeros((n_features, n_tasks));
    // the residuals: `Y - XW` (since W=0, this is just `Y` for now),
    // the residuals are updated during the algorithm as the parameters change
    let mut r = y.to_owned();
    let mut n_steps = 0u32;
    let norm_cols_x = x.map_axis(Axis(0), |col| col.dot(&col));
    let mut gap = F::one() + tol;
    let d_w_tol = tol;
    let tol = tol * y.iter().map(|&y_ij| y_ij * y_ij).sum();
    while n_steps < max_steps {
        let mut w_max = F::zero();
        let mut d_w_max = F::zero();
        for j in 0..n_features {
            if abs_diff_eq!(norm_cols_x[j], F::zero()) {
                continue;
            }
            let mut old_w_j = w.slice_mut(s![j, ..]);
            let x_j = x.slice(s![.., j]);
            let norm_old_w_j = old_w_j.dot(&old_w_j).sqrt();
            if abs_diff_ne!(norm_old_w_j, F::zero()) {
                // r += outer(x_j, old_w_j)
                general_mat_mul(
                    F::one(),
                    &x_j.view().insert_axis(Axis(1)),
                    &old_w_j.view().insert_axis(Axis(0)),
                    F::one(),
                    &mut r,
                );
            }
            let tmp = x_j.dot(&r);
            old_w_j.assign(
                &(block_soft_thresholding(tmp.view(), n_samples * l1_ratio * penalty)
                    / (norm_cols_x[j] + n_samples * (F::one() - l1_ratio) * penalty)),
            );
            let norm_w_j = old_w_j.dot(&old_w_j).sqrt();
            if abs_diff_ne!(norm_w_j, F::zero()) {
                // r -= outer(x_j, old_w_j)
                general_mat_mul(
                    -F::one(),
                    &x_j.insert_axis(Axis(1)),
                    &old_w_j.insert_axis(Axis(0)),
                    F::one(),
                    &mut r,
                );
            }
            let d_w_j = (norm_w_j - norm_old_w_j).abs();
            d_w_max = F::max(d_w_max, d_w_j);
            w_max = F::max(w_max, norm_w_j);
        }
        n_steps += 1;

        if n_steps == max_steps - 1 || abs_diff_eq!(w_max, F::zero()) || d_w_max / w_max < d_w_tol {
            // We've hit one potential stopping criteria
            // check duality gap for ultimate stopping criterion
            gap = duality_gap_mtl(x.view(), y.view(), w.view(), r.view(), l1_ratio, penalty);
            if gap < tol {
                break;
            }
        }
    }

    (w, gap, n_steps)
}

// Algorithm based off of this post: https://math.stackexchange.com/questions/2045579/deriving-block-soft-threshold-from-l-2-norm-prox-operator
fn block_soft_thresholding<F: Float>(x: ArrayView1<F>, threshold: F) -> Array1<F> {
    let norm_x = x.dot(&x).sqrt();
    if norm_x < threshold {
        return Array1::<F>::zeros(x.len());
    }
    let scale = F::one() - threshold / norm_x;
    &x * scale
}

fn duality_gap<'a, F: Float>(
    x: ArrayView2<'a, F>,
    y: ArrayView1<'a, F>,
    w: ArrayView1<'a, F>,
    r: ArrayView1<'a, F>,
    l1_ratio: F,
    penalty: F,
) -> F {
    let half = F::cast(0.5);
    let n_samples = F::cast(x.nrows());
    let l1_reg = l1_ratio * penalty * n_samples;
    let l2_reg = (F::one() - l1_ratio) * penalty * n_samples;
    let xta = x.t().dot(&r) - &w * l2_reg;

    let dual_norm_xta = xta.norm_max();
    let r_norm2 = r.dot(&r);
    let w_norm2 = w.dot(&w);
    let (const_, mut gap) = if dual_norm_xta > l1_reg {
        let const_ = l1_reg / dual_norm_xta;
        let a_norm2 = r_norm2 * const_ * const_;
        (const_, half * (r_norm2 + a_norm2))
    } else {
        (F::one(), r_norm2)
    };
    let l1_norm = w.norm_l1();
    gap += l1_reg * l1_norm - const_ * r.dot(&y)
        + half * l2_reg * (F::one() + const_ * const_) * w_norm2;
    gap
}

fn duality_gap_mtl<'a, F: Float>(
    x: ArrayView2<'a, F>,
    y: ArrayView2<'a, F>,
    w: ArrayView2<'a, F>,
    r: ArrayView2<'a, F>,
    l1_ratio: F,
    penalty: F,
) -> F {
    let half = F::cast(0.5);
    let n_samples = F::cast(x.nrows());
    let l1_reg = l1_ratio * penalty * n_samples;
    let l2_reg = (F::one() - l1_ratio) * penalty * n_samples;
    let xta = x.t().dot(&r) - &w * l2_reg;

    let dual_norm_xta = xta.map_axis(Axis(1), |x| x.dot(&x).sqrt()).norm_max();
    let r_norm2 = r.iter().map(|&rij| rij * rij).sum();
    let w_norm2 = w.iter().map(|&wij| wij * wij).sum();
    let (const_, mut gap) = if dual_norm_xta > l1_reg {
        let const_ = l1_reg / dual_norm_xta;
        let a_norm2 = r_norm2 * const_ * const_;
        (const_, half * (r_norm2 + a_norm2))
    } else {
        (F::one(), r_norm2)
    };
    let rty = r.t().dot(&y);
    let trace_rty = rty.diag().sum();
    let l21_norm = w.map_axis(Axis(1), |wj| (wj.dot(&wj)).sqrt()).sum();
    gap += l1_reg * l21_norm - const_ * trace_rty
        + half * l2_reg * (F::one() + const_ * const_) * w_norm2;
    gap
}

fn variance_params<F: Float, T: AsTargets<Elem = F>, D: Data<Elem = F>>(
    ds: &DatasetBase<ArrayBase<D, Ix2>, T>,
    y_est: ArrayView<F, T::Ix>,
) -> Result<Array1<F>> {
    let nfeatures = ds.nfeatures();
    let nsamples = ds.nsamples();

    let target = ds.targets().as_targets();
    let ndim = target.ndim();

    let ntasks: usize = match ndim {
        1 => 1,
        2 => *target.shape().last().unwrap(),
        _ => {
            return Err(ElasticNetError::IncorrectTargetShape);
        }
    };

    let y_est = y_est.as_targets();

    // check that we have enough samples
    if nsamples < nfeatures + 1 {
        return Err(ElasticNetError::NotEnoughSamples);
    }

    let var_target =
        (&target - &y_est).mapv(|x| x * x).sum() / F::cast(ntasks * (nsamples - nfeatures));

    // `A.t * A` always produces a symmetric matrix
    let ds2 = ds.records().t().dot(ds.records()).with_lapack();
    #[cfg(feature = "blas")]
    let inv_cov = ds2.invh_into();
    #[cfg(not(feature = "blas"))]
    let inv_cov = (|| ds2.qr_into()?.inverse())();

    match inv_cov {
        Ok(inv_cov) => Ok(inv_cov.without_lapack().diag().mapv(|x| var_target * x)),
        Err(_) => Err(ElasticNetError::IllConditioned),
    }
}

/// Compute the intercept as the mean of `y` along each column and center `y` if an intercept
/// should be used, use 0 as intercept and leave `y` unchanged otherwise.
/// If `y` is 2D, mean is 1D and center is 2D. If `y` is 1D, mean is a number and center is 1D.
fn compute_intercept<F: Float, I: RemoveAxis>(
    with_intercept: bool,
    y: ArrayView<F, I>,
) -> (Array<F, I::Smaller>, CowArray<F, I>)
where
    I::Smaller: Dimension<Larger = I>,
{
    if with_intercept {
        let y_mean = y
            // Take the mean of each column (1D array counts as 1 column)
            .mean_axis(Axis(0))
            .expect("Axis 0 length of 0");
        // Subtract y_mean from each "row" of y
        let y_centered = &y - &y_mean.view().insert_axis(Axis(0));
        (y_mean, y_centered.into())
    } else {
        (Array::zeros(y.raw_dim().remove_axis(Axis(0))), y.into())
    }
}

#[cfg(test)]
mod tests {
    use super::{block_coordinate_descent, coordinate_descent, ElasticNet, MultiTaskElasticNet};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, s, Array, Array1, Array2, Axis};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_xoshiro::Xoshiro256Plus;

    use crate::{ElasticNetError, ElasticNetParams, ElasticNetValidParams};
    use linfa::{
        metrics::SingleTargetRegression,
        traits::{Fit, Predict},
        Dataset,
    };

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<ElasticNet<f64>>();
        has_autotraits::<ElasticNetParams<f64>>();
        has_autotraits::<ElasticNetValidParams<f64>>();
        has_autotraits::<ElasticNetError>();
    }

    fn elastic_net_objective(
        x: &Array2<f64>,
        y: &Array1<f64>,
        intercept: f64,
        beta: &Array1<f64>,
        alpha: f64,
        lambda: f64,
    ) -> f64 {
        squared_error(x, y, intercept, beta) + lambda * elastic_net_penalty(beta, alpha)
    }

    fn elastic_net_multi_task_objective(
        x: &Array2<f64>,
        y: &Array2<f64>,
        intercept: &Array1<f64>,
        beta: &Array2<f64>,
        alpha: f64,
        lambda: f64,
    ) -> f64 {
        squared_error_mtl(x, y, intercept, beta) + lambda * elastic_net_mtl_penalty(beta, alpha)
    }

    fn squared_error(x: &Array2<f64>, y: &Array1<f64>, intercept: f64, beta: &Array1<f64>) -> f64 {
        let mut resid = -x.dot(beta);
        resid -= intercept;
        resid += y;
        let mut result = 0.0;
        for r in &resid {
            result += r * r;
        }
        result /= 2.0 * y.len() as f64;
        result
    }

    fn squared_error_mtl(
        x: &Array2<f64>,
        y: &Array2<f64>,
        intercept: &Array1<f64>,
        beta: &Array2<f64>,
    ) -> f64 {
        let mut resid = x.dot(beta);
        resid = &resid * -1.;
        resid = &resid - intercept + y;
        let mut datafit = resid.iter().map(|rij| rij * rij).sum();
        datafit /= 2.0 * x.nrows() as f64;
        datafit
    }

    fn elastic_net_penalty(beta: &Array1<f64>, alpha: f64) -> f64 {
        let mut penalty = 0.0;
        for beta_j in beta {
            penalty += (1.0 - alpha) / 2.0 * beta_j * beta_j + alpha * beta_j.abs();
        }
        penalty
    }

    fn elastic_net_mtl_penalty(beta: &Array2<f64>, alpha: f64) -> f64 {
        let frob_norm: f64 = beta.iter().map(|beta_ij| beta_ij * beta_ij).sum();
        let l21_norm = beta
            .map_axis(Axis(1), |beta_j| (beta_j.dot(&beta_j)).sqrt())
            .sum();
        (1.0 - alpha) / 2.0 * frob_norm + alpha * l21_norm
    }

    #[test]
    fn elastic_net_penalty_works() {
        let beta = array![-2.0, 1.0];
        assert_abs_diff_eq!(
            elastic_net_penalty(&beta, 0.8),
            0.4 + 0.1 + 1.6 + 0.8,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(elastic_net_penalty(&beta, 1.0), 3.0);
        assert_abs_diff_eq!(elastic_net_penalty(&beta, 0.0), 2.5);

        let beta2 = array![0.0, 0.0];
        assert_abs_diff_eq!(elastic_net_penalty(&beta2, 0.8), 0.0);
        assert_abs_diff_eq!(elastic_net_penalty(&beta2, 1.0), 0.0);
        assert_abs_diff_eq!(elastic_net_penalty(&beta2, 0.0), 0.0);
    }

    #[test]
    fn elastic_net_mtl_penalty_works() {
        let beta = array![[-2.0, 1.0, 3.0], [3.0, 1.5, -1.7]];
        assert_abs_diff_eq!(
            elastic_net_mtl_penalty(&beta, 0.7),
            9.472383565516601,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            elastic_net_mtl_penalty(&beta, 1.0),
            7.501976522166574,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            elastic_net_mtl_penalty(&beta, 0.2),
            12.756395304433315,
            epsilon = 1e-12
        );

        let beta2 = array![[0., 0.], [0., 0.], [0., 0.]];
        assert_abs_diff_eq!(elastic_net_mtl_penalty(&beta2, 0.8), 0.0);
        assert_abs_diff_eq!(elastic_net_mtl_penalty(&beta2, 1.2), 0.0);
        assert_abs_diff_eq!(elastic_net_mtl_penalty(&beta2, 0.8), 0.0);
    }

    #[test]
    fn squared_error_works() {
        let x = array![[2.0, 1.0], [-1.0, 2.0]];
        let y = array![1.0, 1.0];
        let beta = array![0.0, 1.0];
        assert_abs_diff_eq!(squared_error(&x, &y, 0.0, &beta), 0.25);
    }

    #[test]
    fn squared_error_mtl_works() {
        let x = array![[1.2, 2.3], [-1.3, 0.3], [-1.3, 0.1]];
        let y = array![
            [0.2, 1.0, 0.0, 1.],
            [-0.3, 0.7, 0.1, 2.],
            [-0.3, 0.7, 2.3, 3.]
        ];
        let beta = array![[2.3, 4.5, 1.2, -3.4], [1.2, -3.4, 0.7, -1.2]];
        assert_abs_diff_eq!(
            squared_error_mtl(&x, &y, &array![0., 0., 0., 0.], &beta),
            41.66298333333333
        );
        let intercept = array![1., 3., 2., 0.3];
        assert_abs_diff_eq!(
            squared_error_mtl(&x, &y, &intercept, &beta),
            29.059983333333335
        );
    }

    #[test]
    fn coordinate_descent_lowers_objective() {
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let y = array![1.0, -1.0];
        let beta = array![0.0, 0.0];
        let intercept = 0.0;
        let alpha = 0.8;
        let lambda = 0.001;
        let objective_start = elastic_net_objective(&x, &y, intercept, &beta, alpha, lambda);
        let opt_result = coordinate_descent(x.view(), y.view(), 1e-4, 3, alpha, lambda);
        let objective_end = elastic_net_objective(&x, &y, intercept, &opt_result.0, alpha, lambda);
        assert!(objective_start > objective_end);
    }

    #[test]
    fn block_coordinate_descent_lowers_objective() {
        let x = array![[1.0, 0., -0.3, 3.2], [0.3, 1.2, -0.6, 1.2]];
        let y = array![[0.3, -1.2, 0.7], [1.4, -3.2, 0.2]];
        let beta = array![[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]];
        let intercept = array![0., 0., 0.];
        let alpha = 0.4;
        let lambda = 0.002;
        let objective_start =
            elastic_net_multi_task_objective(&x, &y, &intercept, &beta, alpha, lambda);
        let opt_result = block_coordinate_descent(x.view(), y.view(), 1e-4, 3, alpha, lambda);
        let objective_end =
            elastic_net_multi_task_objective(&x, &y, &intercept, &opt_result.0, alpha, lambda);
        assert!(objective_start > objective_end);
    }

    #[test]
    fn lasso_zero_works() {
        let dataset = Dataset::from((array![[0.], [0.], [0.]], array![0., 0., 0.]));

        let model = ElasticNet::params()
            .l1_ratio(1.0)
            .penalty(0.1)
            .fit(&dataset)
            .unwrap();

        assert_abs_diff_eq!(model.intercept(), 0.);
        assert_abs_diff_eq!(model.hyperplane(), &array![0.]);
    }

    #[test]
    fn mtl_lasso_zero_works() {
        let dataset = Dataset::from((array![[0.], [0.], [0.]], array![[0.], [0.], [0.]]));

        let model = MultiTaskElasticNet::params()
            .l1_ratio(1.0)
            .penalty(0.1)
            .fit(&dataset)
            .unwrap();

        assert_abs_diff_eq!(model.intercept(), &array![0.]);
        assert_abs_diff_eq!(model.hyperplane(), &array![[0.]]);
    }

    #[test]
    fn lasso_toy_example_works() {
        // Test Lasso on a toy example for various values of alpha.
        // When validating this against glmnet notice that glmnet divides it
        // against n_samples.
        let dataset = Dataset::new(array![[-1.0], [0.0], [1.0]], array![-1.0, 0.0, 1.0]);

        // input for prediction
        let t = array![[2.0], [3.0], [4.0]];
        let model = ElasticNet::lasso().penalty(1e-8).fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.hyperplane(), &array![1.0], epsilon = 1e-6);
        assert_abs_diff_eq!(model.predict(&t), array![2.0, 3.0, 4.0], epsilon = 1e-6);
        assert_abs_diff_eq!(model.duality_gap(), 0.0);

        let model = ElasticNet::lasso().penalty(0.1).fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.hyperplane(), &array![0.85], epsilon = 1e-6);
        assert_abs_diff_eq!(model.predict(&t), array![1.7, 2.55, 3.4], epsilon = 1e-6);
        assert_abs_diff_eq!(model.duality_gap(), 0.0);

        let model = ElasticNet::lasso().penalty(0.5).fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.hyperplane(), &array![0.25], epsilon = 1e-6);
        assert_abs_diff_eq!(model.predict(&t), array![0.5, 0.75, 1.0], epsilon = 1e-6);
        assert_abs_diff_eq!(model.duality_gap(), 0.0);

        let model = ElasticNet::lasso().penalty(1.0).fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.hyperplane(), &array![0.0], epsilon = 1e-6);
        assert_abs_diff_eq!(model.predict(&t), array![0.0, 0.0, 0.0], epsilon = 1e-6);
        assert_abs_diff_eq!(model.duality_gap(), 0.0);
    }

    #[test]
    fn multitask_lasso_toy_example_works() {
        // Test MultiTaskLasso on a toy example for various values of alpha.
        // When validating this against sklearn notice that sklearn divides it
        // against n_samples.
        let dataset = Dataset::new(
            array![[-1.0], [0.0], [1.0]],
            array![[-1.0, 1.0], [0.0, -1.5], [1.0, 1.3]],
        );

        // no intercept fitting
        let t = array![[2.0], [3.0], [4.0]];
        let model = MultiTaskElasticNet::lasso()
            .with_intercept(false)
            .penalty(0.01)
            .fit(&dataset)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), &array![0., 0.]);
        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![[0.9851659, 0.1477748]],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            model.predict(&t),
            array![
                [1.9703319, 0.2955497],
                [2.9554978, 0.4433246],
                [3.9406638, 0.5910995]
            ],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0, epsilon = 1e-9);

        // input for prediction
        let t = array![[2.0], [3.0], [4.0]];
        let model = MultiTaskElasticNet::lasso()
            .penalty(1e-8)
            .fit(&dataset)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), &array![0., 0.2666666667], epsilon = 1e-6);
        assert_abs_diff_eq!(model.hyperplane(), &array![[1., 0.15]], epsilon = 1e-6);
        assert_abs_diff_eq!(
            model.predict(&t),
            array![
                [1.99999997, 0.56666666],
                [2.99999996, 0.71666666],
                [3.99999994, 0.86666666]
            ],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0, epsilon = 1e-9);

        let model = MultiTaskElasticNet::lasso()
            .penalty(0.1)
            .fit(&dataset)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), &array![0., 0.2666666667], epsilon = 1e-6);
        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![[0.851659, 0.127749]],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            model.predict(&t),
            &array![
                [1.70331909, 0.52216453],
                [2.55497864, 0.64991346],
                [3.40663819, 0.77766239]
            ],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0, epsilon = 1e-9);

        let model = MultiTaskElasticNet::lasso()
            .penalty(0.5)
            .fit(&dataset)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), &array![0., 0.2666666667], epsilon = 1e-6);
        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![[0.258298, 0.038744]],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            model.predict(&t),
            &array![
                [0.51659547, 0.34415599],
                [0.77489321, 0.38290065],
                [1.03319094, 0.42164531]
            ],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0, epsilon = 1e-6);

        let model = MultiTaskElasticNet::lasso()
            .penalty(1.0)
            .fit(&dataset)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), &array![0., 0.2666666667], epsilon = 1e-6);
        assert_abs_diff_eq!(model.hyperplane(), &array![[0.0, 0.0]], epsilon = 1e-6);
        assert_abs_diff_eq!(
            model.predict(&t),
            &array![[0., 0.2666666667], [0., 0.2666666667], [0., 0.2666666667]],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn elastic_net_toy_example_works() {
        let dataset = Dataset::new(array![[-1.0], [0.0], [1.0]], array![-1.0, 0.0, 1.0]);

        // for predictions
        let t = array![[2.0], [3.0], [4.0]];
        let model = ElasticNet::params()
            .l1_ratio(0.3)
            .penalty(0.5)
            .fit(&dataset)
            .unwrap();

        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.hyperplane(), &array![0.50819], epsilon = 1e-3);
        assert_abs_diff_eq!(
            model.predict(&t),
            array![1.0163, 1.5245, 2.0327],
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0);

        let model = ElasticNet::params()
            .l1_ratio(0.5)
            .penalty(0.5)
            .fit(&dataset)
            .unwrap();

        assert_abs_diff_eq!(model.intercept(), 0.0);
        assert_abs_diff_eq!(model.hyperplane(), &array![0.45454], epsilon = 1e-3);
        assert_abs_diff_eq!(
            model.predict(&t),
            array![0.9090, 1.3636, 1.8181],
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0);
    }

    #[test]
    fn multitask_elasticnet_toy_example_works() {
        // Test MultiTaskElasticNet on a toy example for various values of alpha
        // and l1_ratio. When validating this against sklearn notice that sklearn
        // divides it against n_samples.
        let dataset = Dataset::new(
            array![[-1.0], [0.0], [1.0]],
            array![[-1.0, 1.0], [0.0, -1.5], [1.0, 1.3]],
        );

        // no intercept fitting
        let t = array![[2.0], [3.0], [4.0]];
        let model = MultiTaskElasticNet::params()
            .with_intercept(false)
            .l1_ratio(0.3)
            .penalty(0.1)
            .fit(&dataset)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), &array![0., 0.]);
        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![[0.86470395, 0.12970559]],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            model.predict(&t),
            array![
                [1.7294079, 0.25941118],
                [2.59411185, 0.38911678],
                [3.4588158, 0.51882237]
            ],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0, epsilon = 1e-12);

        // input for prediction
        let t = array![[2.0], [3.0], [4.0]];
        let model = MultiTaskElasticNet::params()
            .l1_ratio(0.3)
            .penalty(0.1)
            .fit(&dataset)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), &array![0., 0.26666666], epsilon = 1e-6);
        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![[0.86470395, 0.12970559]],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            model.predict(&t),
            array![
                [1.7294079, 0.52607785],
                [2.59411185, 0.65578344],
                [3.4588158, 0.78548904]
            ],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0, epsilon = 1e-12);

        let model = MultiTaskElasticNet::params()
            .l1_ratio(0.5)
            .penalty(0.1)
            .fit(&dataset)
            .unwrap();
        assert_abs_diff_eq!(model.intercept(), &array![0., 0.2666666], epsilon = 1e-6);
        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![[0.861237, 0.12918555]],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            model.predict(&t),
            &array![
                [1.722474, 0.52503777],
                [2.583711, 0.65422332],
                [3.44494799, 0.78340887]
            ],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(model.duality_gap(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn elastic_net_2d_toy_example_works() {
        let dataset = Dataset::new(array![[1.0, 0.0], [0.0, 1.0]], array![3.0, 2.0]);

        let model = ElasticNet::params().penalty(0.0).fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.intercept(), 2.5);
        assert_abs_diff_eq!(model.hyperplane(), &array![0.5, -0.5], epsilon = 0.001);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn elastic_net_diabetes_1_works_like_sklearn() {
        // test that elastic net implementation gives very similar results to
        // sklearn implementation for the first 20 lines taken from the diabetes
        // dataset in linfa/datasets/diabetes_(data|target).csv.gz
        #[rustfmt::skip]
        let x = array![
            [3.807590643342410180e-02, 5.068011873981870252e-02, 6.169620651868849837e-02, 2.187235499495579841e-02, -4.422349842444640161e-02, -3.482076283769860309e-02, -4.340084565202689815e-02, -2.592261998182820038e-03, 1.990842087631829876e-02, -1.764612515980519894e-02],
            [-1.882016527791040067e-03, -4.464163650698899782e-02, -5.147406123880610140e-02, -2.632783471735180084e-02, -8.448724111216979540e-03, -1.916333974822199970e-02, 7.441156407875940126e-02, -3.949338287409189657e-02, -6.832974362442149896e-02, -9.220404962683000083e-02],
            [8.529890629667830071e-02, 5.068011873981870252e-02, 4.445121333659410312e-02, -5.670610554934250001e-03, -4.559945128264750180e-02, -3.419446591411950259e-02, -3.235593223976569732e-02, -2.592261998182820038e-03, 2.863770518940129874e-03, -2.593033898947460017e-02],
            [-8.906293935226029801e-02, -4.464163650698899782e-02, -1.159501450521270051e-02, -3.665644679856060184e-02, 1.219056876180000040e-02, 2.499059336410210108e-02, -3.603757004385269719e-02, 3.430885887772629900e-02, 2.269202256674450122e-02, -9.361911330135799444e-03],
            [5.383060374248070309e-03, -4.464163650698899782e-02, -3.638469220447349689e-02, 2.187235499495579841e-02, 3.934851612593179802e-03, 1.559613951041610019e-02, 8.142083605192099172e-03, -2.592261998182820038e-03, -3.199144494135589684e-02, -4.664087356364819692e-02],
            [-9.269547780327989928e-02, -4.464163650698899782e-02, -4.069594049999709917e-02, -1.944209332987930153e-02, -6.899064987206669775e-02, -7.928784441181220555e-02, 4.127682384197570165e-02, -7.639450375000099436e-02, -4.118038518800790082e-02, -9.634615654166470144e-02],
            [-4.547247794002570037e-02, 5.068011873981870252e-02, -4.716281294328249912e-02, -1.599922263614299983e-02, -4.009563984984299695e-02, -2.480001206043359885e-02, 7.788079970179680352e-04, -3.949338287409189657e-02, -6.291294991625119570e-02, -3.835665973397880263e-02],
            [6.350367559056099842e-02, 5.068011873981870252e-02, -1.894705840284650021e-03, 6.662967401352719310e-02, 9.061988167926439408e-02, 1.089143811236970016e-01, 2.286863482154040048e-02, 1.770335448356720118e-02, -3.581672810154919867e-02, 3.064409414368320182e-03],
            [4.170844488444359899e-02, 5.068011873981870252e-02, 6.169620651868849837e-02, -4.009931749229690007e-02, -1.395253554402150001e-02, 6.201685656730160021e-03, -2.867429443567860031e-02, -2.592261998182820038e-03, -1.495647502491130078e-02, 1.134862324403770016e-02],
            [-7.090024709716259699e-02, -4.464163650698899782e-02, 3.906215296718960200e-02, -3.321357610482440076e-02, -1.257658268582039982e-02, -3.450761437590899733e-02, -2.499265663159149983e-02, -2.592261998182820038e-03, 6.773632611028609918e-02, -1.350401824497050006e-02],
            [-9.632801625429950054e-02, -4.464163650698899782e-02, -8.380842345523309422e-02, 8.100872220010799790e-03, -1.033894713270950005e-01, -9.056118903623530669e-02, -1.394774321933030074e-02, -7.639450375000099436e-02, -6.291294991625119570e-02, -3.421455281914410201e-02],
            [2.717829108036539862e-02, 5.068011873981870252e-02, 1.750591148957160101e-02, -3.321357610482440076e-02, -7.072771253015849857e-03, 4.597154030400080194e-02, -6.549067247654929980e-02, 7.120997975363539678e-02, -9.643322289178400675e-02, -5.906719430815229877e-02],
            [1.628067572730669890e-02, -4.464163650698899782e-02, -2.884000768730720157e-02, -9.113481248670509197e-03, -4.320865536613589623e-03, -9.768885894535990141e-03, 4.495846164606279866e-02, -3.949338287409189657e-02, -3.075120986455629965e-02, -4.249876664881350324e-02],
            [5.383060374248070309e-03, 5.068011873981870252e-02, -1.894705840284650021e-03, 8.100872220010799790e-03, -4.320865536613589623e-03, -1.571870666853709964e-02, -2.902829807069099918e-03, -2.592261998182820038e-03, 3.839324821169769891e-02, -1.350401824497050006e-02],
            [4.534098333546320025e-02, -4.464163650698899782e-02, -2.560657146566450160e-02, -1.255635194240680048e-02, 1.769438019460449832e-02, -6.128357906048329537e-05, 8.177483968693349814e-02, -3.949338287409189657e-02, -3.199144494135589684e-02, -7.563562196749110123e-02],
            [-5.273755484206479882e-02, 5.068011873981870252e-02, -1.806188694849819934e-02, 8.040115678847230274e-02, 8.924392882106320368e-02, 1.076617872765389949e-01, -3.971920784793980114e-02, 1.081111006295440019e-01, 3.605579008983190309e-02, -4.249876664881350324e-02],
            [-5.514554978810590376e-03, -4.464163650698899782e-02, 4.229558918883229851e-02, 4.941532054484590319e-02, 2.457414448561009990e-02, -2.386056667506489953e-02, 7.441156407875940126e-02, -3.949338287409189657e-02, 5.227999979678119719e-02, 2.791705090337660150e-02],
            [7.076875249260000666e-02, 5.068011873981870252e-02, 1.211685112016709989e-02, 5.630106193231849965e-02, 3.420581449301800248e-02, 4.941617338368559792e-02, -3.971920784793980114e-02, 3.430885887772629900e-02, 2.736770754260900093e-02, -1.077697500466389974e-03],
            [-3.820740103798660192e-02, -4.464163650698899782e-02, -1.051720243133190055e-02, -3.665644679856060184e-02, -3.734373413344069942e-02, -1.947648821001150138e-02, -2.867429443567860031e-02, -2.592261998182820038e-03, -1.811826730789670159e-02, -1.764612515980519894e-02],
            [-2.730978568492789874e-02, -4.464163650698899782e-02, -1.806188694849819934e-02, -4.009931749229690007e-02, -2.944912678412469915e-03, -1.133462820348369975e-02, 3.759518603788870178e-02, -3.949338287409189657e-02, -8.944018957797799166e-03, -5.492508739331759815e-02]
        ];
        #[rustfmt::skip]
        let y = array![1.51e+02, 7.5e+01, 1.41e+02, 2.06e+02, 1.35e+02, 9.7e+01, 1.38e+02, 6.3e+01, 1.1e+02, 3.1e+02, 1.01e+02, 6.9e+01, 1.79e+02, 1.85e+02, 1.18e+02, 1.71e+02, 1.66e+02, 1.44e+02, 9.7e+01, 1.68e+02];
        let model = ElasticNet::params()
            .l1_ratio(0.2)
            .penalty(0.5)
            .fit(&Dataset::new(x, y))
            .unwrap();

        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![
                -2.00558969,
                -0.92208413,
                1.27586213,
                -0.06617076,
                0.26484338,
                -0.48702845,
                -0.60274235,
                0.3975141,
                4.33229135,
                1.11981207
            ],
            epsilon = 0.01
        );
        assert_abs_diff_eq!(model.intercept(), 141.283952, epsilon = 1e-1);
        assert!(
            f64::abs(model.duality_gap()) < 1e-4,
            "Duality gap too large"
        );
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn elastic_net_diabetes_2_works_like_sklearn() {
        // test that elastic net implementation gives very similar results to
        // sklearn implementation for the last 20 lines taken from the diabetes
        // dataset in linfa/datasets/diabetes_(data|target).csv.gz
        #[rustfmt::skip]
        let x = array![
            [-7.816532399920170238e-02,5.068011873981870252e-02,7.786338762690199478e-02,5.285819123858220142e-02,7.823630595545419397e-02,6.444729954958319795e-02,2.655027262562750096e-02,-2.592261998182820038e-03,4.067226371449769728e-02,-9.361911330135799444e-03],
            [9.015598825267629943e-03,5.068011873981870252e-02,-3.961812842611620034e-02,2.875809638242839833e-02,3.833367306762140020e-02,7.352860494147960002e-02,-7.285394808472339667e-02,1.081111006295440019e-01,1.556684454070180086e-02,-4.664087356364819692e-02],
            [1.750521923228520000e-03,5.068011873981870252e-02,1.103903904628619932e-02,-1.944209332987930153e-02,-1.670444126042380101e-02,-3.819065120534880214e-03,-4.708248345611389801e-02,3.430885887772629900e-02,2.405258322689299982e-02,2.377494398854190089e-02],
            [-7.816532399920170238e-02,-4.464163650698899782e-02,-4.069594049999709917e-02,-8.141376581713200000e-02,-1.006375656106929944e-01,-1.127947298232920004e-01,2.286863482154040048e-02,-7.639450375000099436e-02,-2.028874775162960165e-02,-5.078298047848289754e-02],
            [3.081082953138499989e-02,5.068011873981870252e-02,-3.422906805671169922e-02,4.367720260718979675e-02,5.759701308243719842e-02,6.883137801463659611e-02,-3.235593223976569732e-02,5.755656502954899917e-02,3.546193866076970125e-02,8.590654771106250032e-02],
            [-3.457486258696700065e-02,5.068011873981870252e-02,5.649978676881649634e-03,-5.670610554934250001e-03,-7.311850844667000526e-02,-6.269097593696699999e-02,-6.584467611156170040e-03,-3.949338287409189657e-02,-4.542095777704099890e-02,3.205915781821130212e-02],
            [4.897352178648269744e-02,5.068011873981870252e-02,8.864150836571099701e-02,8.728689817594480205e-02,3.558176735121919981e-02,2.154596028441720101e-02,-2.499265663159149983e-02,3.430885887772629900e-02,6.604820616309839409e-02,1.314697237742440128e-01],
            [-4.183993948900609910e-02,-4.464163650698899782e-02,-3.315125598283080038e-02,-2.288496402361559975e-02,4.658939021682820258e-02,4.158746183894729970e-02,5.600337505832399948e-02,-2.473293452372829840e-02,-2.595242443518940012e-02,-3.835665973397880263e-02],
            [-9.147093429830140468e-03,-4.464163650698899782e-02,-5.686312160821060252e-02,-5.042792957350569760e-02,2.182223876920789951e-02,4.534524338042170144e-02,-2.867429443567860031e-02,3.430885887772629900e-02,-9.918957363154769225e-03,-1.764612515980519894e-02],
            [7.076875249260000666e-02,5.068011873981870252e-02,-3.099563183506899924e-02,2.187235499495579841e-02,-3.734373413344069942e-02,-4.703355284749029946e-02,3.391354823380159783e-02,-3.949338287409189657e-02,-1.495647502491130078e-02,-1.077697500466389974e-03],
            [9.015598825267629943e-03,-4.464163650698899782e-02,5.522933407540309841e-02,-5.670610554934250001e-03,5.759701308243719842e-02,4.471894645684260094e-02,-2.902829807069099918e-03,2.323852261495349888e-02,5.568354770267369691e-02,1.066170822852360034e-01],
            [-2.730978568492789874e-02,-4.464163650698899782e-02,-6.009655782985329903e-02,-2.977070541108809906e-02,4.658939021682820258e-02,1.998021797546959896e-02,1.222728555318910032e-01,-3.949338287409189657e-02,-5.140053526058249722e-02,-9.361911330135799444e-03],
            [1.628067572730669890e-02,-4.464163650698899782e-02,1.338730381358059929e-03,8.100872220010799790e-03,5.310804470794310353e-03,1.089891258357309975e-02,3.023191042971450082e-02,-3.949338287409189657e-02,-4.542095777704099890e-02,3.205915781821130212e-02],
            [-1.277963188084970010e-02,-4.464163650698899782e-02,-2.345094731790270046e-02,-4.009931749229690007e-02,-1.670444126042380101e-02,4.635943347782499856e-03,-1.762938102341739949e-02,-2.592261998182820038e-03,-3.845911230135379971e-02,-3.835665973397880263e-02],
            [-5.637009329308430294e-02,-4.464163650698899782e-02,-7.410811479030500470e-02,-5.042792957350569760e-02,-2.496015840963049931e-02,-4.703355284749029946e-02,9.281975309919469896e-02,-7.639450375000099436e-02,-6.117659509433449883e-02,-4.664087356364819692e-02],
            [4.170844488444359899e-02,5.068011873981870252e-02,1.966153563733339868e-02,5.974393262605470073e-02,-5.696818394814720174e-03,-2.566471273376759888e-03,-2.867429443567860031e-02,-2.592261998182820038e-03,3.119299070280229930e-02,7.206516329203029904e-03],
            [-5.514554978810590376e-03,5.068011873981870252e-02,-1.590626280073640167e-02,-6.764228304218700139e-02,4.934129593323050011e-02,7.916527725369119917e-02,-2.867429443567860031e-02,3.430885887772629900e-02,-1.811826730789670159e-02,4.448547856271539702e-02],
            [4.170844488444359899e-02,5.068011873981870252e-02,-1.590626280073640167e-02,1.728186074811709910e-02,-3.734373413344069942e-02,-1.383981589779990050e-02,-2.499265663159149983e-02,-1.107951979964190078e-02,-4.687948284421659950e-02,1.549073015887240078e-02],
            [-4.547247794002570037e-02,-4.464163650698899782e-02,3.906215296718960200e-02,1.215130832538269907e-03,1.631842733640340160e-02,1.528299104862660025e-02,-2.867429443567860031e-02,2.655962349378539894e-02,4.452837402140529671e-02,-2.593033898947460017e-02],
            [-4.547247794002570037e-02,-4.464163650698899782e-02,-7.303030271642410587e-02,-8.141376581713200000e-02,8.374011738825870577e-02,2.780892952020790065e-02,1.738157847891100005e-01,-3.949338287409189657e-02,-4.219859706946029777e-03,3.064409414368320182e-03]
        ];
        #[rustfmt::skip]
        let y = array![2.33e+02, 9.1e+01, 1.11e+02, 1.52e+02, 1.2e+02, 6.70e+01, 3.1e+02, 9.4e+01, 1.83e+02, 6.6e+01, 1.73e+02, 7.2e+01, 4.9e+01, 6.4e+01, 4.8e+01, 1.78e+02, 1.04e+02, 1.32e+02, 2.20e+02, 5.7e+01];
        let model = ElasticNet::params()
            .l1_ratio(0.2)
            .penalty(0.5)
            .fit(&Dataset::new(x, y))
            .unwrap();

        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![
                0.19879313,
                1.46970138,
                5.58097318,
                3.80089794,
                1.46466565,
                1.42327857,
                -3.86944632,
                2.60836423,
                4.79584768,
                3.03232988
            ],
            epsilon = 0.01
        );
        assert_abs_diff_eq!(model.intercept(), 126.279, epsilon = 1e-1);
        assert_abs_diff_eq!(model.duality_gap(), 0.00011079, epsilon = 1e-4);
    }

    #[test]
    fn select_subset() {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);

        // check that we are selecting the subsect of informative features
        let mut w = Array::random_using(50, Uniform::new(1., 2.), &mut rng);
        w.slice_mut(s![10..]).fill(0.0);

        let x = Array::random_using((100, 50), Uniform::new(-1., 1.), &mut rng);
        let y = x.dot(&w);
        let train = Dataset::new(x, y);

        let model = ElasticNet::lasso()
            .penalty(0.1)
            .max_iterations(1000)
            .tolerance(1e-10)
            .fit(&train)
            .unwrap();

        // check that we set the last 40 parameters to zero
        let num_zeros = model
            .hyperplane()
            .into_iter()
            .filter(|x| **x < 1e-5)
            .count();

        assert_eq!(num_zeros, 40);

        // predict a small testing dataset
        let x = Array::random_using((100, 50), Uniform::new(-1., 1.), &mut rng);
        let y = x.dot(&w);

        let predicted = model.predict(&x);
        let rms = y.mean_squared_error(&predicted);
        assert!(rms.unwrap() < 0.67);
    }

    #[test]
    fn diabetes_z_score() {
        let dataset = linfa_datasets::diabetes();
        let model = ElasticNet::params().penalty(0.0).fit(&dataset).unwrap();

        // BMI and BP (blood pressure) should be relevant
        let z_score = model.z_score().unwrap();
        assert!(z_score[2] > 2.0);
        assert!(z_score[3] > 2.0);

        // confidence level
        let confidence_level = model.confidence_95th().unwrap();
        assert!(confidence_level[2].0 < 416.);
        assert!(confidence_level[3].0 < 220.);
    }
}
