//! Generalized Linear Models (GLM)

mod distribution;
mod hyperparams;
mod link;

use crate::error::{LinearError, Result};
use crate::float::Float;
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminL1Norm, ArgminL2Norm, ArgminMinMax, ArgminMul, ArgminSignum,
    ArgminSub, ArgminZero,
};
use distribution::TweedieDistribution;
pub use hyperparams::TweedieRegressorParams;
pub use hyperparams::TweedieRegressorValidParams;
use linfa::dataset::AsSingleTargets;
pub use link::Link;

use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{array, concatenate, s};
use ndarray::{Array, Array1, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use serde::{Deserialize, Serialize};

use linfa::traits::*;
use linfa::DatasetBase;

impl<F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = F>>
    Fit<ArrayBase<D, Ix2>, T, LinearError<F>> for TweedieRegressorValidParams<F>
where
    Array1<F>: ArgminAdd<Array1<F>, Array1<F>>
        + ArgminSub<Array1<F>, Array1<F>>
        + ArgminSub<F, Array1<F>>
        + ArgminAdd<F, Array1<F>>
        + ArgminMul<F, Array1<F>>
        + ArgminMul<Array1<F>, Array1<F>>
        + ArgminDot<Array1<F>, F>
        + ArgminL2Norm<F>
        + ArgminL1Norm<F>
        + ArgminSignum
        + ArgminMinMax,
    F: ArgminMul<Array1<F>, Array1<F>> + ArgminZero,
{
    type Object = TweedieRegressor<F>;

    fn fit(&self, ds: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object, F> {
        let (x, y) = (ds.records(), ds.as_single_targets());

        let dist = TweedieDistribution::new(self.power())?;
        let link = self.link();

        // If link is not set we automatically select an appropriate
        // link function

        if !dist.in_range(&y) {
            // An error is sent when y has values in the range not applicable
            // for the distribution
            return Err(LinearError::InvalidTargetRange(self.power()));
        }
        // We initialize the coefficients and intercept
        let mut coef = Array::zeros(x.ncols());
        if self.fit_intercept() {
            let temp = link.link(&array![y.mean().unwrap()]);
            coef = concatenate!(Axis(0), temp, coef);
        }

        // Constructing a struct that satisfies the requirements of the L-BFGS solver
        // with functions implemented for the objective function and the parameter
        // gradient
        let problem = TweedieProblem {
            x: x.view(),
            y,
            fit_intercept: self.fit_intercept(),
            link: &link,
            dist,
            alpha: self.alpha(),
        };
        let linesearch = MoreThuenteLineSearch::new();

        // L-BFGS maintains a history of the past m updates of the
        // position x and gradient ∇f(x), where generally the history
        // size m can be small (often m < 10)
        // For our problem we set m as 7
        let solver = LBFGS::new(linesearch, 7).with_tolerance_grad(F::cast(self.tol()))?;

        let mut result = Executor::new(problem, solver)
            .configure(|state| state.param(coef).max_iters(self.max_iter() as u64))
            .run()?;
        coef = result.state.take_best_param().unwrap();

        if self.fit_intercept() {
            Ok(TweedieRegressor {
                coef: coef.slice(s![1..]).to_owned(),
                intercept: *coef.get(0).unwrap(),
                link,
            })
        } else {
            Ok(TweedieRegressor {
                coef: coef.to_owned(),
                intercept: F::cast(0.),
                link,
            })
        }
    }
}

struct TweedieProblem<'a, F: Float> {
    x: ArrayView2<'a, F>,
    y: ArrayView1<'a, F>,
    fit_intercept: bool,
    link: &'a Link,
    dist: TweedieDistribution<F>,
    alpha: F,
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

impl<'a, A: Float> CostFunction for TweedieProblem<'a, A> {
    type Param = Array1<A>;
    type Output = A;

    // This function calculates the value of the objective function we are trying
    // to minimize,
    //
    // 0.5 * (deviance(y, ypred) + alpha * |p|_2)
    //
    // - `p` is the parameter we are optimizing (coefficients and intercept)
    // - `alpha` is the regularization hyperparameter
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let (ypred, _, offset) = self.ypred(p);

        let dev = self.dist.deviance(self.y, ypred.view())?;

        let pscaled = p
            .slice(s![offset..])
            .mapv(|x| x * A::from(self.alpha).unwrap());

        let obj = A::from(0.5).unwrap() * (dev + p.slice(s![offset..]).dot(&pscaled));

        Ok(obj)
    }
}

impl<'a, A: Float> Gradient for TweedieProblem<'a, A> {
    type Param = Array1<A>;
    type Gradient = Array1<A>;

    fn gradient(&self, p: &Self::Param) -> std::result::Result<Self::Param, argmin::core::Error> {
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

        Ok(objp)
    }
}

/// Generalized Linear Model (GLM) with a Tweedie distribution
///
/// The Regressor can be used to model different GLMs depending on
/// [`power`](TweedieRegressorParams),
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
/// let model = TweedieRegressor::params().fit(&dataset).unwrap();
/// let pred = model.predict(&dataset);
/// let r2 = pred.r2(&dataset).unwrap();
/// println!("r2 from prediction: {}", r2);
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TweedieRegressor<A> {
    /// Estimated coefficients for the linear predictor
    pub coef: Array1<A>,
    /// Intercept or bias added to the linear model
    pub intercept: A,
    link: Link,
}

impl<A: Float, D: Data<Elem = A>> PredictInplace<ArrayBase<D, Ix2>, Array1<A>>
    for TweedieRegressor<A>
{
    /// Predict the target
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<A>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let ypred = x.dot(&self.coef) + self.intercept;
        *y = self.link.inverse(&ypred);
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<A> {
        Array1::zeros(x.nrows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm::hyperparams::TweedieRegressorParams;
    use approx::assert_abs_diff_eq;
    use linfa::Dataset;
    use ndarray::{array, Array2};

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<TweedieRegressor<f64>>();
        has_autotraits::<TweedieRegressorValidParams<f64>>();
        has_autotraits::<TweedieRegressorParams<f64>>();
    }

    macro_rules! test_tweedie {
        ($($name:ident: {power: $power:expr, intercept: $intercept:expr,},)*) => {
            $(
                #[test]
                fn $name() {
                    let coef = array![0.2, -0.1];
                    let mut x: Array2<f64> = array![[1., 1., 1., 1., 1.], [0., 1., 2., 3., 4.]].reversed_axes();
                    let y = x.dot(&coef).mapv(|x| x.exp());

                    let glm = TweedieRegressor::params()
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
