//! Support Vector Regression
use linfa::{dataset::DatasetBase, traits::Fit, traits::Predict, traits::Transformer};
use linfa_kernel::Kernel;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};

use super::permutable_kernel::PermutableKernelRegression;
use super::solver_smo::SolverState;
use super::SolverParams;
use super::{Float, Svm, SvmParams};

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

    let mut sum = c * nu * F::from(target.len()).unwrap() / F::from(2.0).unwrap();
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
impl<'a, F: Float> Fit<'a, Array2<F>, &Array1<F>> for SvmParams<F, F> {
    type Object = Svm<F, F>;

    fn fit(&self, dataset: &DatasetBase<Array2<F>, &Array1<F>>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match (self.c, self.nu) {
            (Some((c, eps)), _) => fit_epsilon(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                c,
                eps,
            ),
            (None, Some((nu, eps))) => fit_nu(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                nu,
                eps,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, Array2<F>, Array1<F>> for SvmParams<F, F> {
    type Object = Svm<F, F>;

    fn fit(&self, dataset: &DatasetBase<Array2<F>, Array1<F>>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match (self.c, self.nu) {
            (Some((c, eps)), _) => fit_epsilon(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                c,
                eps,
            ),
            (None, Some((nu, eps))) => fit_nu(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                nu,
                eps,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, ArrayView2<'a, F>, ArrayView1<'a, F>> for SvmParams<F, F> {
    type Object = Svm<F, F>;

    fn fit(&self, dataset: &DatasetBase<ArrayView2<'a, F>, ArrayView1<'a, F>>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match (self.c, self.nu) {
            (Some((c, eps)), _) => fit_epsilon(
                self.solver_params.clone(),
                *dataset.records(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                c,
                eps,
            ),
            (None, Some((nu, eps))) => fit_nu(
                self.solver_params.clone(),
                *dataset.records(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                nu,
                eps,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, ArrayView2<'a, F>, &'a [F]> for SvmParams<F, F> {
    type Object = Svm<F, F>;

    fn fit(&self, dataset: &DatasetBase<ArrayView2<'a, F>, &'a [F]>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match (self.c, self.nu) {
            (Some((c, eps)), _) => fit_epsilon(
                self.solver_params.clone(),
                *dataset.records(),
                kernel,
                dataset.targets(),
                c,
                eps,
            ),
            (None, Some((nu, eps))) => fit_nu(
                self.solver_params.clone(),
                *dataset.records(),
                kernel,
                dataset.targets(),
                nu,
                eps,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

/// Predict a probability with a set of observations
impl<D: Data<Elem = f64>> Predict<ArrayBase<D, Ix2>, Vec<f64>> for Svm<f64, f64> {
    fn predict(&self, data: ArrayBase<D, Ix2>) -> Vec<f64> {
        data.outer_iter()
            .map(|data| {
                let val = self.weighted_sum(&data) - self.rho;
                // this is safe because `F` is only implemented for `f32` and `f64`
                val
            })
            .collect()
    }
}

/// Predict a probability with a set of observations
impl<D: Data<Elem = f64>> Predict<&ArrayBase<D, Ix2>, Vec<f64>> for Svm<f64, f64> {
    fn predict(&self, data: &ArrayBase<D, Ix2>) -> Vec<f64> {
        data.outer_iter()
            .map(|data| {
                let val = self.weighted_sum(&data) - self.rho;

                // this is safe because `F` is only implemented for `f32` and `f64`
                val
            })
            .collect()
    }
}

#[cfg(test)]
pub mod tests {
    use super::Svm;

    use linfa::dataset::DatasetBase;
    use linfa::metrics::Regression;
    use linfa::traits::{Fit, Predict};
    use ndarray::{Array, Array1};

    #[test]
    fn test_linear_epsilon_regression() {
        let target = Array::linspace(0f64, 10., 100);
        let mut sin_curve = Array::zeros((100, 1));
        for (i, val) in target.iter().enumerate() {
            sin_curve[(i, 0)] = *val;
        }

        let dataset = DatasetBase::new(sin_curve.view(), target.view());

        let model = Svm::params()
            .nu_eps(2., 0.01)
            .gaussian_kernel(50.)
            .fit(&dataset);

        println!("{}", model);

        let predicted = Array1::from(model.predict(sin_curve));
        assert!(predicted.mean_squared_error(&target.view()) < 1e-2);
    }

    #[test]
    fn test_linear_nu_regression() {
        let target = Array::linspace(0f64, 10., 100);
        let mut sin_curve = Array::zeros((100, 1));
        for (i, val) in target.iter().enumerate() {
            sin_curve[(i, 0)] = *val;
        }

        let dataset = DatasetBase::new(sin_curve.view(), target.view());

        let model = Svm::params()
            .nu_eps(2., 0.01)
            .gaussian_kernel(50.)
            .fit(&dataset);

        println!("{}", model);

        let predicted = Array1::from(model.predict(sin_curve));
        assert!(predicted.mean_squared_error(&target.view()) < 1e-2);
    }

    #[test]
    fn test_regression_linear_kernel() {
        // simple 2d straight line
        let targets = Array::linspace(0f64, 10., 100);
        let records = targets.clone().into_shape((100, 1)).unwrap();

        let dataset = (records, targets).into();

        // Test the precomputed dot product in the linear kernel case
        let model = Svm::params().nu_eps(2., 0.01).linear_kernel().fit(&dataset);

        println!("{}", model);

        let predicted = Array1::from(model.predict(dataset.records()));
        assert!(predicted.mean_squared_error(&dataset.targets().view()) < 1e-2);
    }
}
