//! Support Vector Regression
use linfa::{dataset::DatasetBase, traits::Fit, traits::Predict};
use ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix2};
use std::ops::Mul;

use super::permutable_kernel::{Kernel, PermutableKernelRegression};
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
pub fn fit_epsilon<'a, A: Float>(
    params: SolverParams<A>,
    kernel: &'a Kernel<'a, A>,
    target: &'a [A],
    c: A,
    p: A,
) -> Svm<'a, A, A> {
    let mut linear_term = vec![A::zero(); 2 * target.len()];
    let mut targets = vec![true; 2 * target.len()];

    for i in 0..target.len() {
        linear_term[i] = p - target[i];
        targets[i] = true;

        linear_term[i + target.len()] = p + target[i];
        targets[i + target.len()] = false;
    }

    let kernel = PermutableKernelRegression::new(kernel);
    let solver = SolverState::new(
        vec![A::zero(); 2 * target.len()],
        linear_term,
        targets.to_vec(),
        kernel,
        vec![c; 2 * target.len()],
        params,
        false,
    );

    let mut res = solver.solve();

    for i in 0..target.len() {
        let tmp = res.alpha[i + target.len()];
        res.alpha[i] -= tmp;
    }
    res.alpha.truncate(target.len());

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
pub fn fit_nu<'a, A: Float>(
    params: SolverParams<A>,
    kernel: &'a Kernel<'a, A>,
    target: &'a [A],
    c: A,
    nu: A,
) -> Svm<'a, A, A> {
    let mut alpha = vec![A::zero(); 2 * target.len()];
    let mut linear_term = vec![A::zero(); 2 * target.len()];
    let mut targets = vec![true; 2 * target.len()];

    let mut sum = c * nu * A::from(target.len()).unwrap() / A::from(2.0).unwrap();
    for i in 0..target.len() {
        alpha[i] = A::min(sum, c);
        alpha[i + target.len()] = A::min(sum, c);
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
        kernel,
        vec![c; 2 * target.len()],
        params,
        false,
    );

    let mut res = solver.solve();

    for i in 0..target.len() {
        let tmp = res.alpha[i + target.len()];
        res.alpha[i] -= tmp;
    }
    res.alpha.truncate(target.len());

    res.with_phantom()
}

/// Regress obserations
///
/// Take a number of observations and project them to optimal continuous targets.
impl<'a, F: Float> Fit<'a, Kernel<'a, F>, &Array1<F>> for SvmParams<F, F> {
    type Object = Svm<'a, F, F>;

    fn fit(&self, dataset: &'a DatasetBase<Kernel<'a, F>, &Array1<F>>) -> Self::Object {
        match (self.c, self.nu) {
            (Some((c, eps)), _) => fit_epsilon(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets().as_slice().unwrap(),
                c,
                eps,
            ),
            (None, Some((nu, eps))) => fit_nu(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets().as_slice().unwrap(),
                nu,
                eps,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, Kernel<'a, F>, ArrayView1<'a, F>> for SvmParams<F, F> {
    type Object = Svm<'a, F, F>;

    fn fit(&self, dataset: &'a DatasetBase<Kernel<'a, F>, ArrayView1<'a, F>>) -> Self::Object {
        match (self.c, self.nu) {
            (Some((c, eps)), _) => fit_epsilon(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets().as_slice().unwrap(),
                c,
                eps,
            ),
            (None, Some((nu, eps))) => fit_nu(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets().as_slice().unwrap(),
                nu,
                eps,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, D: Data<Elem = f64>> Predict<ArrayBase<D, Ix2>, Vec<f64>> for Svm<'a, f64, f64> {
    fn predict(&self, data: ArrayBase<D, Ix2>) -> Vec<f64> {
        data.outer_iter()
            .map(|data| {
                let val = match self.linear_decision {
                    Some(ref x) => x.mul(&data).sum() - self.rho,
                    None => self.kernel.weighted_sum(&self.alpha, data.view()) - self.rho,
                };

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
    use linfa::traits::{Fit, Predict, Transformer};
    use linfa_kernel::{Kernel, KernelMethod};
    use ndarray::{Array, Array1};

    #[test]
    fn test_linear_epsilon_regression() {
        let target = Array::linspace(0f64, 10., 100);
        let mut sin_curve = Array::zeros((100, 1));
        for (i, val) in target.iter().enumerate() {
            sin_curve[(i, 0)] = *val;
        }

        let kernel = Kernel::params()
            .method(KernelMethod::Gaussian(50.))
            .transform(&sin_curve);

        let dataset = DatasetBase::new(kernel, target.view());

        let model = Svm::params().nu_eps(2., 0.01).fit(&dataset);

        println!("{}", model);

        let predicted = Array1::from(model.predict(sin_curve.clone()));
        assert!(predicted.mean_squared_error(&target.view()) < 1e-2);
    }

    #[test]
    fn test_linear_nu_regression() {
        let target = Array::linspace(0f64, 10., 100);
        let mut sin_curve = Array::zeros((100, 1));
        for (i, val) in target.iter().enumerate() {
            sin_curve[(i, 0)] = *val;
        }

        let kernel = Kernel::params()
            .method(KernelMethod::Gaussian(50.))
            .transform(&sin_curve);

        let dataset = DatasetBase::new(kernel, target.view());

        let model = Svm::params().nu_eps(2., 0.01).fit(&dataset);

        println!("{}", model);

        let predicted = Array1::from(model.predict(sin_curve.clone()));
        assert!(predicted.mean_squared_error(&target.view()) < 1e-2);
    }
}
