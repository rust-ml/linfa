mod distribution;
mod link;

use crate::error::{LinearError, Result};
use crate::{ArgminParam, Float};
use distribution::TweedieDistribution;
use link::Link;

use argmin::core::{ArgminOp, Executor};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{array, s, stack};
use ndarray::{Array, Array1, Array2, Axis};

struct TweedieRegressor {
    alpha: f64,
    fit_intercept: bool,
    power: f64,
    link: Option<Link>,
    max_iter: usize,
    tol: f64,
}

impl TweedieRegressor {
    fn new() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            power: 1.,
            link: None,
            max_iter: 100,
            tol: 1e-4,
        }
    }

    fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    fn power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    fn link(mut self, link: Link) -> Self {
        self.link = Some(link);
        self
    }

    fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

impl TweedieRegressor {
    fn fit<A: Float>(&self, x: &Array2<A>, y: &Array1<A>) -> Result<FittedTweedieRegressor<A>> {
        let dist = TweedieDistribution::new(self.power)?;

        if self.alpha < 0. {
            return Err(LinearError::InvalidValue(format!(
                "Penalty term must be a non-negative number, got: {}",
                self.alpha
            )));
        }

        let link: Link;
        if let Some(value) = self.link {
            link = value;
        } else {
            link = match self.power {
                p if p <= 0. => Link::Identity,
                p if p >= 1. => Link::Log,
                p => {
                    return Err(LinearError::InvalidValue(format!("{}", p)));
                }
            }
        }

        if !dist.in_range(&y) {
            return Err(LinearError::InvalidValue(format!(
                "Some value(s) of y are out of the valid range for power value {}",
                self.power
            )));
        }

        let mut coef: Array1<A>;
        if self.fit_intercept {
            coef = Array::zeros(x.ncols() + 1);
            let temp = *link.link(&array![y.mean().unwrap()]).get(0).unwrap();
            let element = coef.get_mut(0).unwrap();
            *element = temp;
        } else {
            coef = Array::zeros(x.ncols());
        }

        let problem = TweedieProblem {
            x,
            y,
            fit_intercept: self.fit_intercept,
            link: &link,
            dist,
            alpha: self.alpha,
        };
        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 7).with_tol_grad(A::from(self.tol).unwrap());
        let res = Executor::new(problem, solver, ArgminParam(coef))
            .max_iters(self.max_iter as u64)
            .run()?;
        let coef_res = res.state.get_best_param();
        let coef_res_array = coef_res.as_array();

        if self.fit_intercept {
            Ok(FittedTweedieRegressor {
                coef: coef_res_array.slice(s![1..]).to_owned(),
                intercept: *coef_res_array.get(0).unwrap(),
                link,
            })
        } else {
            Ok(FittedTweedieRegressor {
                coef: coef_res_array.to_owned(),
                intercept: A::from(0.).unwrap(),
                link,
            })
        }
    }
}

struct TweedieProblem<'a, A: Float> {
    x: &'a Array2<A>,
    y: &'a Array1<A>,
    fit_intercept: bool,
    link: &'a Link,
    dist: TweedieDistribution,
    alpha: f64,
}

impl<'a, A: Float> ArgminOp for TweedieProblem<'a, A> {
    type Param = ArgminParam<A>;
    type Output = A;
    type Hessian = ();
    type Jacobian = Array1<A>;
    type Float = A;

    fn apply(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let p = p.as_array();
        let obj;
        if self.fit_intercept {
            let lin_pred = self
                .x
                .view()
                .dot(&p.slice(s![1..]))
                .mapv(|x| x + *p.get(0).unwrap());
            let ypred = self.link.inverse(&lin_pred);

            let dev = self.dist.deviance(self.y, &ypred)?;
            let p_scaled = p.slice(s![1..]).mapv(|x| x * A::from(self.alpha).unwrap());
            obj = A::from(0.5).unwrap() * dev
                + A::from(0.5).unwrap() * p.slice(s![1..]).dot(&p_scaled);
        } else {
            let lin_pred = self.x.dot(p);
            let ypred = self.link.inverse(&lin_pred);

            let dev = self.dist.deviance(self.y, &ypred);
            let p_scaled = p.mapv(|x| x * A::from(self.alpha).unwrap());
            obj = A::from(0.5).unwrap() * dev.unwrap() + A::from(0.5).unwrap() * p.dot(&p_scaled);
        }
        Ok(obj)
    }

    fn gradient(&self, p: &Self::Param) -> std::result::Result<Self::Param, argmin::core::Error> {
        let p = p.as_array();
        let mut objp;
        if self.fit_intercept {
            let lin_pred = self
                .x
                .view()
                .dot(&p.slice(s![1..]))
                .mapv(|x| x + *p.get(0).unwrap());
            let ypred = self.link.inverse(&lin_pred);

            let der = self.link.inverse_derviative(&lin_pred);

            let temp = der * self.dist.deviance_derivative(self.y, &ypred);
            let devp = stack![Axis(0), array![temp.sum()], temp.dot(self.x)];
            let p_scaled = p.slice(s![1..]).mapv(|x| x * A::from(self.alpha).unwrap());
            objp = devp.mapv(|x| x * A::from(0.5).unwrap());
            objp.slice_mut(s![1..])
                .zip_mut_with(&p_scaled, |x, y| *x += *y);
        } else {
            let lin_pred = self.x.dot(p);
            let ypred = self.link.inverse(&lin_pred);

            let der = self.link.inverse_derviative(&lin_pred);

            let temp = der * self.dist.deviance_derivative(self.y, &ypred);
            let devp = temp.dot(self.x);
            let p_scaled = p.mapv(|x| x * A::from(self.alpha).unwrap());
            objp = devp.mapv(|x| x * A::from(0.5).unwrap());
            objp.zip_mut_with(&p_scaled, |x, y| *x += *y);
        }
        Ok(ArgminParam(objp))
    }
}

struct FittedTweedieRegressor<A> {
    coef: Array1<A>,
    intercept: A,
    link: Link,
}

impl<A: Float> FittedTweedieRegressor<A> {
    fn predict(&self, x: &Array2<A>) -> Array1<A> {
        let ypred = x.dot(&self.coef) + self.intercept;
        self.link.inverse(&ypred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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
                        let glm = glm.fit(&x, &y).unwrap();

                        assert_abs_diff_eq!(glm.intercept, coef.get(0).unwrap(), epsilon = 1e-3);
                        assert_abs_diff_eq!(glm.coef, coef.slice(s![1..]), epsilon = 1e-3);
                    } else {
                        let glm = glm.fit(&x, &y).unwrap();

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
