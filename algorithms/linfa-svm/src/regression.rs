//! Support Vector Regression
use linfa::{
    dataset::{AsSingleTargets, DatasetBase},
    traits::Fit,
    traits::Transformer,
    traits::{Predict, PredictInplace},
};
use linfa_kernel::Kernel;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};

use super::error::{Result, SvmError};
use super::permutable_kernel::PermutableKernelRegression;
use super::solver_smo::SolverState;
use super::SolverParams;
use super::{Float, Svm, SvmValidParams};

/// Support Vector Regression with epsilon tolerance
///
/// This methods solves a binary SVC problem with a penalizing parameter epsilon between (0, inf). This defines the margin of tolerance, where no penalty is given to errors.
///
/// # Parameters
///
/// * `params` - Solver parameters (threshold etc.)
/// * `kernel` - the kernel matrix `Q`
/// * `targets` - the continuous targets `y_i`
/// * `c` - C value for all targets
/// * `p` - epsilon value for all targets
pub fn fit_epsilon<F: Float>(
    params: SolverParams<F>,
    dataset: ArrayView2<F>,
    kernel: Kernel<F>,
    target: &[F],
    c: F,
    p: F,
) -> Svm<F, F> {
    let mut linear_term = vec![F::zero(); 2 * target.len()];
    let mut targets = vec![true; 2 * target.len()];

    for i in 0..target.len() {
        linear_term[i] = p - target[i];
        targets[i] = true;

        linear_term[i + target.len()] = p + target[i];
        targets[i + target.len()] = false;
    }

    let kernel = PermutableKernelRegression::new(kernel);
    let solver = SolverState::new(
        vec![F::zero(); 2 * target.len()],
        linear_term,
        targets.to_vec(),
        dataset,
        kernel,
        vec![c; 2 * target.len()],
        params,
        false,
    );

    let res = solver.solve();

    res.with_phantom()
}

/// Support Vector Regression with nu parameter
///
/// This methods solves a binary SVC problem with parameter nu, defining how many support vectors should be used. This parameter should be in range (0, 1).
///
/// # Parameters
///
/// * `params` - Solver parameters (threshold etc.)
/// * `kernel` - the kernel matrix `Q`
/// * `targets` - the continuous targets `y_i`
/// * `c` - C value for all targets
/// * `nu` - nu value for all targets
pub fn fit_nu<F: Float>(
    params: SolverParams<F>,
    dataset: ArrayView2<F>,
    kernel: Kernel<F>,
    target: &[F],
    c: F,
    nu: F,
) -> Svm<F, F> {
    let mut alpha = vec![F::zero(); 2 * target.len()];
    let mut linear_term = vec![F::zero(); 2 * target.len()];
    let mut targets = vec![true; 2 * target.len()];

    let mut sum = c * nu * F::cast(target.len()) / F::cast(2.0);
    for i in 0..target.len() {
        alpha[i] = F::min(sum, c);
        alpha[i + target.len()] = F::min(sum, c);
        sum -= alpha[i];

        linear_term[i] = -target[i];
        targets[i] = true;

        linear_term[i + target.len()] = target[i];
        targets[i + target.len()] = false;
    }

    let kernel = PermutableKernelRegression::new(kernel);
    let solver = SolverState::new(
        alpha,
        linear_term,
        targets.to_vec(),
        dataset,
        kernel,
        vec![c; 2 * target.len()],
        params,
        false,
    );

    let res = solver.solve();

    res.with_phantom()
}

/// Regress observations
///
/// Take a number of observations and project them to optimal continuous targets.
macro_rules! impl_regression {
    ($records:ty, $targets:ty, $f:ty) => {
        impl Fit<$records, $targets, SvmError> for SvmValidParams<$f, $f> {
            type Object = Svm<$f, $f>;

            fn fit(&self, dataset: &DatasetBase<$records, $targets>) -> Result<Self::Object> {
                let kernel = self.kernel_params().transform(dataset.records());
                let target = dataset.try_single_target()?;
                let target = target.as_slice().unwrap();

                let ret = match (self.c(), self.nu()) {
                    (Some((c, eps)), _) => fit_epsilon(
                        self.solver_params().clone(),
                        dataset.records().view(),
                        kernel,
                        target,
                        c,
                        eps,
                    ),
                    (None, Some((nu, eps))) => fit_nu(
                        self.solver_params().clone(),
                        dataset.records().view(),
                        kernel,
                        target,
                        nu,
                        eps,
                    ),
                    _ => panic!("Set either C value or Nu value"),
                };

                Ok(ret)
            }
        }
    };
}

// impl_regression!(Array2<f32>, Array2<f32>, f32);
// impl_regression!(Array2<f64>, Array2<f64>, f64);
// impl_regression!(ArrayView2<'_, f32>, ArrayView2<'_, f32>, f32);
// impl_regression!(ArrayView2<'_, f64>, ArrayView2<'_, f64>, f64);
impl_regression!(Array2<f32>, Array1<f32>, f32);
impl_regression!(Array2<f64>, Array1<f64>, f64);
impl_regression!(ArrayView2<'_, f32>, ArrayView1<'_, f32>, f32);
impl_regression!(ArrayView2<'_, f64>, ArrayView1<'_, f64>, f64);

macro_rules! impl_predict {
    ( $($t:ty),* ) => {
    $(
        /// Predict a probability with a feature vector
        impl Predict<Array1<$t>, $t> for Svm<$t, $t> {
            fn predict(&self, data: Array1<$t>) -> $t {
                self.weighted_sum(&data) - self.rho
            }
        }
        /// Predict a probability with a feature vector
        impl<'a> Predict<ArrayView1<'a, $t>, $t> for Svm<$t, $t> {
            fn predict(&self, data: ArrayView1<'a, $t>) -> $t {
                self.weighted_sum(&data) - self.rho
            }
        }

        /// Classify observations
        ///
        /// This function takes a number of features and predicts target probabilities that they belong to
        /// the positive class.
        impl<D: Data<Elem = $t>> PredictInplace<ArrayBase<D, Ix2>, Array1<$t>> for Svm<$t, $t> {
            fn predict_inplace<'a>(&'a self, data: &ArrayBase<D, Ix2>, targets: &mut Array1<$t>) {
                assert_eq!(data.nrows(), targets.len(), "The number of data points must match the number of output targets.");

                for (data, target) in data.outer_iter().zip(targets.iter_mut()) {
                    *target = self.weighted_sum(&data) - self.rho;
                }
            }

            fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<$t> {
                Array1::zeros(x.nrows())
            }
        }

    ) *
    }
}

impl_predict!(f32, f64);

#[cfg(test)]
pub mod tests {
    use super::Svm;
    use crate::error::Result;

    use linfa::dataset::Dataset;
    use linfa::metrics::SingleTargetRegression;
    use linfa::traits::{Fit, Predict};
    use ndarray::Array;

    #[test]
    fn test_linear_epsilon_regression() -> Result<()> {
        let target = Array::linspace(0f64, 10., 100);
        let mut sin_curve = Array::zeros((100, 1));
        for (i, val) in target.iter().enumerate() {
            sin_curve[(i, 0)] = *val;
        }

        let dataset = Dataset::new(sin_curve, target);

        let model = Svm::params()
            .nu_eps(2., 0.01)
            .gaussian_kernel(50.)
            .fit(&dataset)?;

        println!("{}", model);

        let predicted = model.predict(dataset.records());
        assert!(predicted.mean_squared_error(&dataset).unwrap() < 1e-2);

        Ok(())
    }

    #[test]
    fn test_linear_nu_regression() -> Result<()> {
        let target = Array::linspace(0f64, 10., 100);
        let mut sin_curve = Array::zeros((100, 1));
        for (i, val) in target.iter().enumerate() {
            sin_curve[(i, 0)] = *val;
        }

        let dataset = Dataset::new(sin_curve, target);

        let model = Svm::params()
            .nu_eps(2., 0.01)
            .gaussian_kernel(50.)
            .fit(&dataset)?;

        println!("{}", model);

        let predicted = model.predict(&dataset);
        assert!(predicted.mean_squared_error(&dataset).unwrap() < 1e-2);

        Ok(())
    }

    #[test]
    fn test_regression_linear_kernel() -> Result<()> {
        // simple 2d straight line
        let targets = Array::linspace(0f64, 10., 100);
        let records = targets.clone().into_shape((100, 1)).unwrap();

        let dataset = Dataset::new(records, targets);

        // Test the precomputed dot product in the linear kernel case
        let model = Svm::params()
            .nu_eps(2., 0.01)
            .linear_kernel()
            .fit(&dataset)?;

        println!("{}", model);

        let predicted = model.predict(&dataset);
        assert!(predicted.mean_squared_error(&dataset).unwrap() < 1e-2);

        Ok(())
    }
}
