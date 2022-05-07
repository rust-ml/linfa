//! Implement Platt calibration with Newton method
//!
//! Platt scaling is a way of transforming the outputs of a classification model into a probability
//! distribution over classes. It is, for example, used in the calibration of SVM models to predict
//! plausible probability values.
//!
//! # Example
//!
//! ```rust, ignore
//! let model = ...;
//!
//! let model = Platt::params()
//!      .fit_with(model, &train)?;
//!
//! let pred: Array1<Pr> = model.predict(&valid);
//! ```

use std::marker::PhantomData;

use crate::dataset::{DatasetBase, Pr};
use crate::traits::{FitWith, Predict, PredictInplace};
use crate::{Float, ParamGuard};

use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Data, Ix1, Ix2};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use thiserror::Error;

/// Fitted Platt model
///
/// This model contains a sigmoid scaling parameters and a second univariate, uncalibrated model.
/// The output of the uncalibrated model is scaled with the following function:
/// ```text
/// g(x) = 1 / (1 + exp(A * f(x) + B)
/// ```
///
/// The scaling factors `A` and `B` are estimated with the Newton's method, presented in the
/// following paper: <https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf>
pub struct Platt<F, O> {
    a: F,
    b: F,
    obj: O,
}

/// Parameters for Platt's Newton method
pub struct PlattValidParams<F, O> {
    maxiter: usize,
    minstep: F,
    sigma: F,
    phantom: PhantomData<O>,
}

pub struct PlattParams<F, O>(PlattValidParams<F, O>);

impl<F: Float, O> Default for PlattParams<F, O> {
    fn default() -> Self {
        Self(PlattValidParams {
            maxiter: 100,
            minstep: F::cast(1e-10),
            sigma: F::cast(1e-12),
            phantom: PhantomData,
        })
    }
}

impl<F: Float, O> PlattParams<F, O> {
    /// Set the maximum number of iterations in the optimization process
    ///
    /// The Newton's method is an iterative optimization process, which uses the first and second
    /// order gradients to find optimal `A` and `B`. This function caps the maximal number of
    /// iterations.
    pub fn maxiter(mut self, maxiter: usize) -> Self {
        self.0.maxiter = maxiter;

        self
    }

    /// Set the minimum stepsize in the line search
    ///
    /// After estimating the Hessian matrix, a line search is performed to find the optimal step
    /// size in each optimization step. In each attempt the stepsize is halfed until this threshold
    /// is reached. After reaching the threshold the algorithm fails because the desired precision
    /// could not be achieved.
    pub fn minstep(mut self, minstep: F) -> Self {
        self.0.minstep = minstep;

        self
    }

    /// Set the Hessian's sigma value
    ///
    /// The Hessian matrix is regularized with H' = H + sigma I to avoid numerical issues. This
    /// function set the amount of regularization.
    pub fn sigma(mut self, sigma: F) -> Self {
        self.0.sigma = sigma;

        self
    }
}

impl<F: Float, O> ParamGuard for PlattParams<F, O> {
    type Checked = PlattValidParams<F, O>;
    type Error = PlattError;

    fn check_ref(&self) -> Result<&Self::Checked, PlattError> {
        if self.0.maxiter == 0 {
            Err(PlattError::MaxIterReached)
        } else if self.0.minstep.is_negative() {
            Err(PlattError::MinStepNegative(
                self.0.minstep.to_f32().unwrap(),
            ))
        } else if self.0.sigma.is_negative() {
            Err(PlattError::SigmaNegative(self.0.sigma.to_f32().unwrap()))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, PlattError> {
        self.check_ref()?;
        Ok(self.0)
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Error, Debug, Clone)]
/// Platt Newton's method errors
///
/// Errors occur when setting invalid parameters or the optimization process fails.
pub enum PlattError {
    #[error("line search did not converge")]
    LineSearchNotConverged,
    #[error("platt scaling did not converge")]
    MaxIterReached,
    #[error("maxiter should be larger than zero")]
    MaxIterZero,
    #[error("minstep should be positive, is {0}")]
    MinStepNegative(f32),
    #[error("sigma should be positive, is {0}")]
    SigmaNegative(f32),
    #[error(transparent)]
    LinfaError(#[from] crate::error::Error),
}

impl<F: Float, O> Platt<F, O> {
    /// Create default parameter set for the Platt scaling algorithm
    ///
    /// The default values are:
    /// * `maxiter`: 100,
    /// * `minstep`: 1e-10,
    /// * `sigma`: 1e-12
    ///
    pub fn params() -> PlattParams<F, O> {
        PlattParams::default()
    }
}

impl<'a, F: Float, O: 'a> FitWith<'a, Array2<F>, Array1<bool>, PlattError>
    for PlattValidParams<F, O>
where
    O: PredictInplace<Array2<F>, Array1<F>>,
{
    type ObjectIn = O;
    type ObjectOut = Platt<F, O>;

    /// Calibrate another model with Platt scaling
    ///
    /// This function takes another model and binary decision dataset and calibrates it to produce
    /// probability values. The returned model therefore implements the prediction trait for
    /// probability targets.
    fn fit_with(
        &self,
        obj: O,
        ds: &DatasetBase<Array2<F>, Array1<bool>>,
    ) -> Result<Self::ObjectOut, PlattError> {
        let predicted = obj.predict(ds);

        let (a, b) = platt_newton_method(predicted.view(), ds.targets().view(), self)?;

        Ok(Platt { a, b, obj })
    }
}

impl<F: Float, D, O> PredictInplace<ArrayBase<D, Ix2>, Array1<Pr>> for Platt<F, O>
where
    D: Data<Elem = F>,
    O: PredictInplace<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>,
{
    fn predict_inplace(&self, data: &ArrayBase<D, Ix2>, targets: &mut Array1<Pr>) {
        assert_eq!(
            data.nrows(),
            targets.len(),
            "The number of data points must match the number of output targets."
        );
        for (x, target) in self.obj.predict(data).iter().zip(targets.iter_mut()) {
            *target = platt_predict(*x, self.a, self.b);
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<Pr> {
        Array1::default(x.nrows())
    }
}

/// Predict a probability with the sigmoid function
///
/// Similar to stable sigmoid implementations
/// <https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth>
///
/// # Parameters
/// * `x`: uncalibrated value f(x)
/// * `a`: parameter A,
/// * `b`: parameter B,
pub fn platt_predict<F: Float>(x: F, a: F, b: F) -> Pr {
    let f_apb = a * x + b;
    let f_apb = f_apb.to_f32().unwrap();

    // avoid numerical problems for large f_apb
    if f_apb >= 0.0 {
        Pr::new((-f_apb).exp() / (1.0 + (-f_apb).exp()))
    } else {
        Pr::new(1.0 / (1.0 + f_apb.exp()))
    }
}

/// Run Newton's method to find optimal `A` and `B` values
///
/// The optimization process happens in two steps, first the closed-form Hessian matrix and
/// gradient vectors are calculated. Then a line-search tries to find the optimal learning rate
/// for each step.
//#[allow(clippy::suspicious_operation_groupings)]
pub fn platt_newton_method<'a, F: Float, O>(
    reg_values: ArrayView1<'a, F>,
    labels: ArrayView1<'a, bool>,
    params: &PlattValidParams<F, O>,
) -> Result<(F, F), PlattError> {
    let (num_pos, num_neg) = labels.iter().fold((0, 0), |mut val, x| {
        match x {
            true => val.0 += 1,
            false => val.1 += 1,
        }

        val
    });
    let (num_pos, num_neg) = (num_pos as f32, num_neg as f32);

    let (hi_target, lo_target) = ((num_pos + 1.0) / (num_pos + 2.0), 1.0 / (num_neg + 2.0));

    let target_values = labels
        .iter()
        .map(|x| if *x { hi_target } else { lo_target })
        .map(|x| F::from(x).unwrap())
        .collect::<Vec<_>>();

    let reg_values = reg_values
        .into_iter()
        .map(|x| F::from(*x).unwrap())
        .collect::<Vec<_>>();

    let mut a = F::zero();
    let mut b = F::from((num_neg + 1.0) / (num_pos + 1.0)).unwrap().ln();
    let mut fval = F::zero();

    for (v, t) in reg_values.iter().zip(target_values.iter()) {
        let f_apb = *v * a + b;
        if f_apb >= F::zero() {
            fval += *t * f_apb + (F::one() + (-f_apb).exp()).ln();
        } else {
            fval += (*t - F::one()) * f_apb + (F::one() + f_apb.exp()).ln();
        }
    }

    let mut idx = 0;
    for _ in 0..params.maxiter {
        let (mut h11, mut h22) = (params.sigma, params.sigma);
        let (mut h21, mut g1, mut g2) = (F::zero(), F::zero(), F::zero());

        for (v, t) in reg_values.iter().zip(target_values.iter()) {
            let f_apb = *v * a + b;

            let (p, q) = if f_apb >= F::zero() {
                (
                    (-f_apb).exp() / (F::one() + (-f_apb).exp()),
                    F::one() / (F::one() + (-f_apb).exp()),
                )
            } else {
                (
                    F::one() / (F::one() + f_apb.exp()),
                    f_apb.exp() / (F::one() + f_apb.exp()),
                )
            };

            let d2 = p * q;
            h11 += *v * *v * d2;
            h22 += d2;
            h21 += *v * d2;

            let d1 = *t - p;
            g1 += *v * d1;
            g2 += d1;
        }

        if g1.abs() < F::from(1e-5 * reg_values.len() as f32).unwrap()
            && g2.abs() < F::from(1e-5 * reg_values.len() as f32).unwrap()
        {
            break;
        }

        let det = h11 * h22 - h21.powi(2);
        let d_a = -(h22 * g1 - h21 * g2) / det;
        let d_b = -(-h21 * g1 + h11 * g2) / det;
        let gd = g1 * d_a + g2 * d_b;

        let mut stepsize = F::one();
        while stepsize >= params.minstep {
            let new_a = a + stepsize * d_a;
            let new_b = b + stepsize * d_b;
            let mut newf = F::zero();

            for (v, t) in reg_values.iter().zip(target_values.iter()) {
                let f_apb = *v * new_a + new_b;

                if f_apb >= F::zero() {
                    newf += *t * f_apb + (F::one() + (-f_apb).exp()).ln();
                } else {
                    newf += (*t - F::one()) * f_apb + (F::one() + f_apb.exp()).ln();
                }
            }

            if newf < fval + F::from(1e-4).unwrap() * stepsize * gd {
                a = new_a;
                b = new_b;
                fval = newf;
                break;
            } else {
                stepsize /= F::one() + F::one();
            }

            if stepsize < params.minstep {
                return Err(PlattError::LineSearchNotConverged);
            }
        }

        idx += 1;
    }

    if params.maxiter == idx {
        return Err(PlattError::MaxIterReached);
    }

    Ok((a, b))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    use super::{platt_newton_method, Platt, PlattValidParams};
    use crate::{
        traits::{FitWith, Predict, PredictInplace},
        DatasetBase, Float, ParamGuard,
    };

    /// Generate dummy values which can be predicted with the Platt model
    fn generate_dummy_values<F: Float, R: Rng>(
        a: F,
        b: F,
        n: usize,
        rng: &mut R,
    ) -> (Array1<F>, Array1<bool>) {
        // generate probability values, omit p = 0.0, p = 1.0 to avoid infinity in reverse function
        let prob_values = Array1::linspace(
            F::one() / F::from(n).unwrap(),
            F::one() - F::one() / F::from(n).unwrap(),
            n - 2,
        );

        // generate regression values with inverse function
        let reg_values = prob_values
            .iter()
            .map(|x| (F::one() - *x) / *x)
            .map(|x| (x.ln() - b) / a)
            .collect();

        // roll decision according to probability
        let decisions = prob_values
            .iter()
            .map(|x| rng.gen_bool(x.to_f64().unwrap()))
            .collect();

        (reg_values, decisions)
    }

    macro_rules! test_newton_solver {
        ($($fnc:ident, $a_val:expr);*) => {
            $(
            #[test]
            fn $fnc() {
                let mut rng = SmallRng::seed_from_u64(42);

                let params: PlattValidParams<f32, ()> = PlattValidParams {
                    maxiter: 100,
                    minstep: 1e-10,
                    sigma: 1e-12,
                    phantom: std::marker::PhantomData,
                };

                let a = $a_val as f32;

                for b in &[a / 2.0, a * 2.0] {
                    let (reg_vals, dec_vals) = generate_dummy_values(a, *b, 10000, &mut rng);
                    let (a_est, b_est) = platt_newton_method(reg_vals.view(), dec_vals.view(), &params).unwrap();

                    assert_abs_diff_eq!(a_est, a, epsilon = 0.15);
                    assert_abs_diff_eq!(b_est, b, epsilon = 0.1);
                }
            }
            )*
        };
    }

    // Check solutions for
    // * a = 1.0, (b = 0.5, b = 2.0)
    // * a = 2.0, (b = 1.0, b = 4.0)
    // * a = 5.0, (b = 2.5, b = 10.0)
    test_newton_solver!(
        newton_solver_1, 1;
        newton_solver_2, 2;
        newton_solver_5, 5
    );

    struct DummyModel {
        reg_vals: Array1<f32>,
    }

    impl PredictInplace<Array2<f32>, Array1<f32>> for DummyModel {
        fn predict_inplace(&self, x: &Array2<f32>, y: &mut Array1<f32>) {
            assert_eq!(
                x.nrows(),
                y.len(),
                "The number of data points must match the number of output targets."
            );
            *y = self.reg_vals.clone();
        }

        fn default_target(&self, x: &Array2<f32>) -> Array1<f32> {
            Array1::zeros(x.nrows())
        }
    }

    #[test]
    /// Check that the predicted probabilities are non-decreasing monotonical
    fn ordered_probabilities() {
        let mut rng = SmallRng::seed_from_u64(42);

        let (reg_vals, dec_vals) = generate_dummy_values(1.0, 0.5, 102, &mut rng);
        let records = Array2::zeros((100, 3));
        let dataset = DatasetBase::new(records, dec_vals);

        let model = DummyModel { reg_vals };

        let platt = Platt::params().fit_with(model, &dataset).unwrap();

        let pred_probabilities = platt.predict(&dataset).to_vec();

        for vals in pred_probabilities.windows(2) {
            if vals[0] > vals[1] {
                panic!("Probabilities are not monotonically increasing!");
            }
        }
    }

    #[test]
    #[should_panic]
    fn panic_maxiter_zero() {
        Platt::<f32, ()>::params().maxiter(0).check().unwrap();
    }

    #[test]
    #[should_panic]
    fn panic_minstep_negative() {
        Platt::<f32, ()>::params().minstep(-5.0).check().unwrap();
    }

    #[test]
    #[should_panic]
    fn panic_sigma_negative() {
        Platt::<f32, ()>::params().sigma(-1.0).check().unwrap();
    }
}
