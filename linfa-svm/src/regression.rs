use linfa_kernel::Kernel;
use super::{SvmResult, Float};
use super::SolverParams;
use super::solver_smo::SolverState;
use super::permutable_kernel::PermutableKernelRegression;

pub fn fit_epsilon<'a, A: Float>(
    params: &'a SolverParams<A>,
    kernel: &'a Kernel<A>,
    target: &'a [A],
    c: A,
    p: A,
) -> SvmResult<'a, A> {
    let mut linear_term = vec![A::zero(); 2 * target.len()];
    let mut targets = vec![true; 2 * target.len()];

    for i in 0..target.len() {
        linear_term[i] = p - target[i];
        targets[i] = true;

        linear_term[i + target.len()] = p + target[i];
        targets[i] = false;
    }

    let kernel = PermutableKernelRegression::new(kernel);
    let solver = SolverState::new(
        vec![A::zero(); 2 * target.len()],
        linear_term,
        targets.to_vec(),
        kernel,
        vec![c; target.len()],
        params,
        false,
    );

    let mut res = solver.solve();

    for i in 0..target.len() {
        let tmp = res.alpha[i+target.len()];
        res.alpha[i] -= tmp;
    }
    res.alpha.truncate(target.len());

    res


}

pub fn fit_nu<'a, A: Float>(
    params: &'a SolverParams<A>,
    kernel: &'a Kernel<A>,
    target: &'a [A],
    c: A,
    nu: A,
) -> SvmResult<'a, A> {
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
        targets[i] = false;
    }

    let kernel = PermutableKernelRegression::new(kernel);
    let solver = SolverState::new(
        alpha,
        linear_term,
        targets.to_vec(),
        kernel,
        vec![c; target.len()],
        params,
        false,
    );

    solver.solve()
}

#[cfg(test)]
pub mod tests {
    use super::{SolverParams, fit_nu, fit_epsilon};

    use linfa_kernel::Kernel;
    use ndarray::Array;

    #[test]
    fn test_linear_epsilon_regression() {
        let target = Array::linspace(0., 10., 100);
        let entries = Array::ones((100, 2));
        let kernel = Kernel::gaussian(entries, 100.);

        let params = SolverParams {
            eps: 1e-3,
            shrinking: false
        };

        let svr = fit_epsilon(&params, &kernel, &target.as_slice().unwrap(), 1.0, 0.1);
        println!("{}", svr);
    }

    #[test]
    fn test_linear_nu_regression() {
        let target = Array::linspace(0., 10., 100);
        let entries = Array::ones((100, 2));
        let kernel = Kernel::gaussian(entries, 100.);

        let params = SolverParams {
            eps: 1e-3,
            shrinking: false
        };

        let svr = fit_nu(&params, &kernel, &target.as_slice().unwrap(), 1.0, 0.1);
        println!("{}", svr);
    }
}
