//! Generalized Linear Models (GLM)

mod distribution;
mod link;

use crate::error::{LinearError, Result};
use crate::float::{ArgminParam, Float};
use distribution::TweedieDistribution;
pub use link::Link;

use argmin::core::{ArgminOp, Executor};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{array, concatenate, s};
use ndarray::{Array, Array1, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use serde::{Deserialize, Serialize};

use linfa::traits::*;
use linfa::{dataset::AsTargets, DatasetBase};

/// Generalized Linear Model (GLM) with a Tweedie distribution
///
/// The Regressor can be used to model different GLMs depending on
/// [`power`](struct.TweedieRegressor.html#method.power),
/// which determines the underlying distribution.
///
/// | Power  | Distribution           |
/// | ------ | ---------------------- |
/// | 0      | Normal                 |
/// | 1      | Poisson                |
/// | (1, 2) | Compound Poisson Gamma |
/// | 2      | Gamma                  |
/// | 3      | Inverse Gaussian       |
///
/// NOTE: No distribution exists between 0 and 1
///
/// Learn more from sklearn's excellent [User Guide](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression)
///
/// ## Examples
///
/// Here's an example on how to train a GLM on the `diabetes` dataset
/// ```rust
/// use linfa::traits::{Fit, Predict};
/// use linfa_linear::TweedieRegressor;
/// use linfa::prelude::SingleTargetRegression;
///
/// let dataset = linfa_datasets::diabetes();
/// let model = TweedieRegressor::default().fit(&dataset).unwrap();
/// let pred = model.predict(&dataset);
/// let r2 = pred.r2(&dataset).unwrap();
/// println!("r2 from prediction: {}", r2);
/// ```
#[derive(Serialize, Deserialize)]
pub struct TweedieRegressor {
    alpha: f64,
    fit_intercept: bool,
    power: f64,
    link: Option<Link>,
    max_iter: usize,
    tol: f64,
}

impl Default for TweedieRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl TweedieRegressor {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            power: 1.,
            link: None,
            max_iter: 100,
            tol: 1e-4,
        }
    }

    /// Constant that multiplies with the penalty term and thus determines the
    /// regularization strenght. `alpha` set to 0 is equivalent to unpenalized GLM.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Specifies whether a bias or intercept should be added to the model
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// The power determines the underlying target distribution
    pub fn power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// The link function of the GLM, for mapping from linear predictor `x @ coeff + intercept` to
    /// the prediction. If no value is set, the link will be selected based on the following,
    /// - [`identity`](Link::Identity) for Normal distribution (`power` = 0)
    /// - [`log`](Link::Log) for Poisson, Gamma and Inverse Gaussian distributions (`power` >= 1)
    pub fn link(mut self, link: Link) -> Self {
        self.link = Some(link);
        self
    }

    /// Maximum number of iterations for the LBFGS solver
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Stopping criterion for the LBFGS solver
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

impl<A: Float, D: Data<Elem = A>, T: AsTargets<Elem = A>> Fit<ArrayBase<D, Ix2>, T, LinearError>
    for TweedieRegressor
{
    type Object = FittedTweedieRegressor<A>;

    fn fit(&self, ds: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let (x, y) = (ds.records(), ds.try_single_target()?);

        let dist = TweedieDistribution::new(self.power)?;

        if self.alpha < 0. {
            return Err(linfa::Error::Parameters(format!(
                "Penalty term must be a non-negative number, got: {}",
                self.alpha
            ))
            .into());
        }

        // If link is not set we automatically select an appropriate
        // link function
        let link = match self.link {
            Some(x) => Ok(x),
            None if self.power <= 0. => Ok(Link::Identity),
            None if self.power >= 1. => Ok(Link::Log),
            None => Err(linfa::Error::Parameters(format!(
                "Power value cannot be between 0 and 1, got: {}",
                self.power
            ))),
        }?;

        if !dist.in_range(&y) {
            // An error is sent when y has values in the range not applicable
            // for the distribution
            return Err(linfa::Error::Parameters(format!(
                "Some value(s) of y are out of the valid range for power value {}",
                self.power
            ))
            .into());
        }

        // We initialize the coefficients and intercept
        let mut coef = Array::zeros(x.ncols());
        if self.fit_intercept {
            let temp = link.link(&array![y.mean().unwrap()]);
            coef = concatenate!(Axis(0), temp, coef);
        }

        // Constructing a struct that satisfies the requirements of the L-BFGS solver
        // with functions implemented for the objective function and the parameter
        // gradient
        let problem = TweedieProblem {
            x: x.view(),
            y,
            fit_intercept: self.fit_intercept,
            link: &link,
            dist,
            alpha: self.alpha,
        };
        let linesearch = MoreThuenteLineSearch::new();

        // L-BFGS maintains a history of the past m updates of the
        // position x and gradient ∇f(x), where generally the history
        // size m can be small (often m < 10)
        // For our problem we set m as 7
        let solver = LBFGS::new(linesearch, 7).with_tol_grad(A::from(self.tol).unwrap());

        let result = Executor::new(problem, solver, ArgminParam(coef))
            .max_iters(self.max_iter as u64)
            .run()?;
        coef = result.state.get_best_param().as_array().to_owned();

        if self.fit_intercept {
            Ok(FittedTweedieRegressor {
                coef: coef.slice(s![1..]).to_owned(),
                intercept: *coef.get(0).unwrap(),
                link,
            })
        } else {
            Ok(FittedTweedieRegressor {
                coef: coef.to_owned(),
                intercept: A::from(0.).unwrap(),
                link,
            })
        }
    }
}

struct TweedieProblem<'a, A: Float> {
    x: ArrayView2<'a, A>,
    y: ArrayView1<'a, A>,
    fit_intercept: bool,
    link: &'a Link,
    dist: TweedieDistribution,
    alpha: f64,
}

impl<'a, A: Float> TweedieProblem<'a, A> {
    fn ypred(&self, p: &Array1<A>) -> (Array1<A>, Array1<A>, usize) {
        let mut offset = 0;
        let mut intercept = A::from(0.).unwrap();
        if self.fit_intercept {
            offset = 1;
            intercept = *p.get(0).unwrap();
        }

        let lin_pred = self
            .x
            .view()
            .dot(&p.slice(s![offset..]))
            .mapv(|x| x + intercept);

        (self.link.inverse(&lin_pred), lin_pred, offset)
    }
}

impl<'a, A: Float> ArgminOp for TweedieProblem<'a, A> {
    type Param = ArgminParam<A>;
    type Output = A;
    type Hessian = ();
    type Jacobian = Array1<A>;
    type Float = A;

    // This function calculates the value of the objective function we are trying
    // to minimize,
    //
    // 0.5 * (deviance(y, ypred) + alpha * |p|_2)
    //
    // - `p` is the parameter we are optimizing (coefficients and intercept)
    // - `alpha` is the regularization hyperparameter
    fn apply(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let p = p.as_array();

        let (ypred, _, offset) = self.ypred(p);

        let dev = self.dist.deviance(self.y, ypred.view())?;

        let pscaled = p
            .slice(s![offset..])
            .mapv(|x| x * A::from(self.alpha).unwrap());

        let obj = A::from(0.5).unwrap() * (dev + p.slice(s![offset..]).dot(&pscaled));

        Ok(obj)
    }

    fn gradient(&self, p: &Self::Param) -> std::result::Result<Self::Param, argmin::core::Error> {
        let p = p.as_array();

        let (ypred, lin_pred, offset) = self.ypred(p);

        let devp;
        let der = self.link.inverse_derviative(&lin_pred);
        let temp = der * self.dist.deviance_derivative(self.y, ypred.view());
        if self.fit_intercept {
            devp = concatenate![Axis(0), array![temp.sum()], temp.dot(&self.x)];
        } else {
            devp = temp.dot(&self.x);
        }

        let pscaled = p
            .slice(s![offset..])
            .mapv(|x| x * A::from(self.alpha).unwrap());

        let mut objp = devp.mapv(|x| x * A::from(0.5).unwrap());
        objp.slice_mut(s![offset..])
            .zip_mut_with(&pscaled, |x, y| *x += *y);

        Ok(ArgminParam(objp))
    }
}

/// Fitted Tweedie regressor model for scoring
#[derive(Serialize, Deserialize)]
pub struct FittedTweedieRegressor<A> {
    /// Estimated coefficients for the linear predictor
    pub coef: Array1<A>,
    /// Intercept or bias added to the linear model
    pub intercept: A,
    link: Link,
}

impl<A: Float, D: Data<Elem = A>> PredictRef<ArrayBase<D, Ix2>, Array1<A>>
    for FittedTweedieRegressor<A>
{
    /// Predict the target
    fn predict_ref(&self, x: &ArrayBase<D, Ix2>) -> Array1<A> {
        let ypred = x.dot(&self.coef) + self.intercept;
        self.link.inverse(&ypred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::Dataset;
    use ndarray::{array, Array2};

    macro_rules! test_tweedie {
        ($($name:ident: {power: $power:expr, intercept: $intercept:expr,},)*) => {
            $(
                #[test]
                fn $name() {
                    let coef = array![0.2, -0.1];
                    let mut x: Array2<f64> = array![[1., 1., 1., 1., 1.], [0., 1., 2., 3., 4.]].reversed_axes();
                    let y = x.dot(&coef).mapv(|x| x.exp());

                    let glm = TweedieRegressor::new()
                        .alpha(0.)
                        .power($power)
                        .link(Link::Log)
                        .tol(1e-7)
                        .fit_intercept($intercept);

                    if $intercept {
                        x = x.slice(s![.., 1..]).to_owned();
                        let dataset = Dataset::new(x, y);
                        let glm = glm.fit(&dataset).unwrap();

                        assert_abs_diff_eq!(glm.intercept, coef.get(0).unwrap(), epsilon = 1e-3);
                        assert_abs_diff_eq!(glm.coef, coef.slice(s![1..]), epsilon = 1e-3);
                    } else {
                        let dataset = Dataset::new(x, y);
                        let glm = glm.fit(&dataset).unwrap();

                        assert_abs_diff_eq!(glm.coef, coef, epsilon = 1e-3);
                    }
                }
            )*
        }
    }

    test_tweedie! {
        test_glm_normal1: {
            power: 0.,
            intercept: true,
        },
        test_glm_normal2: {
            power: 0.,
            intercept: false,
        },
        test_glm_poisson1: {
            power: 1.,
            intercept: true,
        },
        test_glm_poisson2: {
            power: 1.,
            intercept: false,
        },
        test_glm_gamma1: {
            power: 2.,
            intercept: true,
        },
        test_glm_gamma2: {
            power: 2.,
            intercept: false,
        },
        test_glm_inverse_gaussian1: {
            power: 3.,
            intercept: true,
        },
        test_glm_inverse_gaussian2: {
            power: 3.,
            intercept: false,
        },
        test_glm_tweedie1: {
            power: 1.5,
            intercept: true,
        },
        test_glm_tweedie2: {
            power: 1.5,
            intercept: false,
        },
    }
}
