///! Support Vector Regression
use std::marker::PhantomData;
use linfa::dataset::Pr;

use super::permutable_kernel::{Kernel, PermutableKernelRegression};
use super::solver_smo::SolverState;
use super::SolverParams;
use super::{Float, Svm};

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
    kernel: &'a Kernel<A>,
    target: &'a [A],
    c: A,
    p: A,
) -> Svm<'a, A, Pr> {
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
    kernel: &'a Kernel<A>,
    target: &'a [A],
    c: A,
    nu: A,
) -> Svm<'a, A, Pr> {
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

#[cfg(test)]
pub mod tests {
    use super::{fit_epsilon, fit_nu, SolverParams};

    use linfa::metrics::Regression;
    use linfa_kernel::Kernel;
    use ndarray::{Array, Array1};

    #[test]
    fn test_linear_epsilon_regression() {
        let target = Array::linspace(0f64, 10., 100).to_vec();
        let mut sin_curve = Array::zeros((100, 1));
        for (i, val) in target.iter().enumerate() {
            sin_curve[(i, 0)] = *val;
        }

        let kernel = Kernel::gaussian(&sin_curve, 50.);

        let params = SolverParams {
            eps: 1e-8,
            shrinking: false,
        };

        let svr = fit_epsilon(&params, &kernel, &target, 2.0, 0.01);
        println!("{}", svr);

        let predicted = sin_curve
            .outer_iter()
            .map(|x| svr.predict(x))
            .collect::<Array1<_>>();

        assert!(predicted.mean_squared_error(&target) < 1e-2);

        //dbg!(&predicted);
    }

    #[test]
    fn test_linear_nu_regression() {
        let target = Array::linspace(0f64, 10., 100).to_vec();
        let mut sin_curve = Array::zeros((100, 1));
        for (i, val) in target.iter().enumerate() {
            sin_curve[(i, 0)] = *val;
        }

        let kernel = Kernel::gaussian(&sin_curve, 50.);

        let params = SolverParams {
            eps: 1e-8,
            shrinking: false,
        };

        let svr = fit_nu(&params, &kernel, &target, 2.0, 1.0);
        println!("{}", svr);

        let predicted = sin_curve
            .outer_iter()
            .map(|x| svr.predict(x))
            .collect::<Array1<_>>();

        assert!(predicted.mean_squared_error(&target) < 1e-2);
    }
}
