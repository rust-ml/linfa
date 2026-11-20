//! # Logistic Regression
//!
//! ## The Big Picture
//!
//! `linfa-logistic` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.
//!
//! ## Current state
//! `linfa-logistic` provides a pure Rust implementation of a [binomial logistic regression model](LogisticRegression) and a [multinomial logistic regression model](MultiLogisticRegression).
//!
//! ## Examples
//!
//! There is an usage example in the `examples/` directory. To run, use:
//!
//! ```bash
//! $ cargo run --example winequality
//! ```
//!

pub mod error;

use crate::error::{Error, Result};
use argmin::core::{CostFunction, Executor, Gradient, IterState, OptimizationResult, Solver};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use linfa::dataset::AsSingleTargets;
use linfa::prelude::DatasetBase;
use linfa::traits::{Fit, PredictInplace};
use ndarray::{
    s, Array, Array1, Array2, ArrayBase, ArrayView, Axis, CowArray, Data, Dimension, IntoDimension,
    Ix1, Ix2, RemoveAxis, Slice, Zip,
};
use ndarray_stats::QuantileExt;
use std::cell::RefCell;
use std::default::Default;

#[cfg(feature = "serde")]
use serde_crate::de::DeserializeOwned;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

mod argmin_param;
mod float;
mod hyperparams;

use argmin_param::*;
use float::Float;
use hyperparams::{LogisticRegressionParams, LogisticRegressionValidParams};

/// A two-class logistic regression model.
///
/// Logistic regression combines linear models with
/// the sigmoid function `sigm(x) = 1/(1+exp(-x))`
/// to learn a family of functions that map the feature space to `[0,1]`.
///
/// Logistic regression is used in binary classification
/// by interpreting the predicted value as the probability that the sample
/// has label `1`. A threshold can be set in the [fitted model](FittedLogisticRegression) to decide the minimum
/// probability needed to classify a sample as `1`, which defaults to `0.5`.
///
/// In this implementation any binary set of labels can be used, not necessarily `0` and `1`.
///
/// l2 regularization is used by this algorithm and is weighted by parameter `alpha`. Setting `alpha`
/// close to zero removes regularization and the problem solved minimizes only the
/// empirical risk. On the other hand, setting `alpha` to a high value increases
/// the weight of the l2 norm of the linear model coefficients in the cost function.
///
/// ## Examples
///
/// Here's an example on how to train a logistic regression model on the `winequality` dataset
/// ```rust
/// use linfa::traits::{Fit, Predict};
/// use linfa_logistic::LogisticRegression;
///
/// // Example on using binary labels different from 0 and 1
/// let dataset = linfa_datasets::winequality().map_targets(|x| if *x > 6 { "good" } else { "bad" });
/// let model = LogisticRegression::default().fit(&dataset).unwrap();
/// let prediction = model.predict(&dataset);
/// ```
pub type LogisticRegression<F> = LogisticRegressionParams<F, Ix1>;

/// Validated version of `LogisticRegression`
pub type ValidLogisticRegression<F> = LogisticRegressionValidParams<F, Ix1>;

/// A multinomial class logistic regression model.
///
/// The output labels can map to any discrete feature space, since the algorithm calculates the
/// likelihood of a feature vector corresponding to any given outcome using the softmax function
/// `softmax(x) = exp(x) / sum(exp(xi))`
///
/// l2 regularization is used by this algorithm and is weighted by parameter `alpha`. Setting `alpha`
/// close to zero removes regularization and the problem solved minimizes only the
/// empirical risk. On the other hand, setting `alpha` to a high value increases
/// the weight of the l2 norm of the linear model coefficients in the cost function.
pub type MultiLogisticRegression<F> = LogisticRegressionParams<F, Ix2>;

/// Validated version of `MultiLogisticRegression`
pub type ValidMultiLogisticRegression<F> = LogisticRegressionValidParams<F, Ix2>;

impl<F: Float, D: Dimension> Default for LogisticRegressionParams<F, D> {
    fn default() -> Self {
        LogisticRegressionParams::new()
    }
}

type LBFGSType<F, D> = LBFGS<
    MoreThuenteLineSearch<ArgminParam<F, D>, ArgminParam<F, D>, F>,
    ArgminParam<F, D>,
    ArgminParam<F, D>,
    F,
>;
type LBFGSType1<F> = LBFGSType<F, Ix1>;
type LBFGSType2<F> = LBFGSType<F, Ix2>;

type IterStateType<F, D> = IterState<ArgminParam<F, D>, ArgminParam<F, D>, (), (), (), F>;

impl<F: Float, D: Dimension> LogisticRegressionValidParams<F, D> {
    /// Create the initial parameters, either from a user supplied array
    /// or an array of 0s
    fn setup_init_params(&self, dims: D::Pattern) -> ArgminParam<F, D> {
        if let Some(params) = self.initial_params.as_ref() {
            ArgminParam(params.clone())
        } else {
            let mut dims = dims.into_dimension();
            dims.as_array_view_mut()[0] += self.fit_intercept as usize;
            ArgminParam(Array::zeros(dims))
        }
    }

    /// Ensure that `x` and `y` have the right shape and that all data and
    /// configuration parameters are finite.
    fn validate_data<A: Data<Elem = F>, B: Data<Elem = F>>(
        &self,
        x: &ArrayBase<A, Ix2>,
        y: &ArrayBase<B, D>,
    ) -> Result<()> {
        if x.shape()[0] != y.shape()[0] {
            return Err(Error::MismatchedShapes(x.shape()[0], y.shape()[0]));
        }
        if x.iter().any(|x| !x.is_finite()) || y.iter().any(|y| !y.is_finite()) {
            return Err(Error::InvalidValues);
        }
        self.validate_init_dims(x.shape()[1], y.shape().get(1).copied())?;
        Ok(())
    }

    fn validate_init_dims(&self, mut n_features: usize, n_classes: Option<usize>) -> Result<()> {
        if let Some(params) = self.initial_params.as_ref() {
            let shape = params.shape();
            n_features += self.fit_intercept as usize;
            if n_features != shape[0] {
                return Err(Error::InitialParameterFeaturesMismatch {
                    n_features,
                    rows: shape[0],
                });
            }
            if let Some(n_classes) = n_classes {
                if n_classes != shape[1] {
                    return Err(Error::InitialParameterClassesMismatch {
                        n_classes,
                        cols: shape[1],
                    });
                }
            }
        }
        Ok(())
    }

    /// Create a `LogisticRegressionProblem`.
    fn setup_problem<'a, A: Data<Elem = F>>(
        &self,
        x: &'a ArrayBase<A, Ix2>,
        target: Array<F, D>,
    ) -> LogisticRegressionProblem<'a, F, A, D> {
        LogisticRegressionProblem::new(x, target, self.alpha, self.offset.clone())
    }

    /// Create the LBFGS solver using MoreThuenteLineSearch and set gradient
    /// tolerance.
    fn setup_solver(&self) -> LBFGSType<F, D> {
        let linesearch = MoreThuenteLineSearch::new();
        LBFGS::new(linesearch, 10)
            .with_tolerance_grad(self.gradient_tolerance)
            .unwrap()
    }
}

impl<
        F: Float,
        #[cfg(feature = "serde")] D: Dimension + Serialize + DeserializeOwned,
        #[cfg(not(feature = "serde"))] D: Dimension,
    > LogisticRegressionValidParams<F, D>
{
    /// Run the LBFGS solver until it converges or runs out of iterations.
    fn run_solver<P: SolvableProblem<F, D>>(
        &self,
        problem: P,
        solver: P::Solver,
        init_params: ArgminParam<F, D>,
    ) -> Result<OptimizationResult<P, P::Solver, IterStateType<F, D>>> {
        Executor::new(problem, solver)
            .configure(|state| state.param(init_params).max_iters(self.max_iterations))
            .run()
            .map_err(move |err| err.into())
    }
}

impl<C: Ord + Clone, F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = C>>
    Fit<ArrayBase<D, Ix2>, T, Error> for ValidLogisticRegression<F>
{
    type Object = FittedLogisticRegression<F, C>;

    /// Given a 2-dimensional feature matrix array `x` with shape
    /// (n_samples, n_features) and an array of target classes to predict,
    /// create a `FittedLinearRegression` object which allows making
    /// predictions.
    ///
    /// The array of target classes `y` must have exactly two discrete values, (e.g. 0 and 1, "cat"
    /// and "dog", ...), which represent the two different classes the model is supposed to
    /// predict.
    ///
    /// The array `y` must also have exactly `n_samples` items, i.e.
    /// exactly as many items as there are rows in the feature matrix `x`.
    ///
    /// This method returns an error if any of the preconditions are violated,
    /// i.e. any values are `Inf` or `NaN`, `y` doesn't have as many items as
    /// `x` has rows, or if other parameters (gradient_tolerance, alpha) have
    /// been set to inalid values.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let (x, y) = (dataset.records(), dataset.targets());
        let (labels, target) = label_classes(y)?;
        self.validate_data(x, &target)?;

        if let Some(ref offset) = self.offset {
            if offset.len() != x.nrows() {
                return Err(Error::OffsetLengthMismatch {
                    offset_len: offset.len(),
                    n_samples: x.nrows(),
                });
            }
        }

        let problem = self.setup_problem(x, target);
        let solver = self.setup_solver();
        let init_params = self.setup_init_params(x.ncols());
        let result = self.run_solver(problem, solver, init_params)?;

        let params = result
            .state
            .best_param
            .unwrap_or(self.setup_init_params(x.ncols()));
        let (w, intercept) = convert_params(x.ncols(), params.as_array());
        Ok(FittedLogisticRegression::new(
            *intercept.view().into_scalar(),
            w.to_owned(),
            labels,
        ))
    }
}

impl<C: Ord + Clone, F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = C>>
    Fit<ArrayBase<D, Ix2>, T, Error> for ValidMultiLogisticRegression<F>
{
    type Object = MultiFittedLogisticRegression<F, C>;

    /// Given a 2-dimensional feature matrix array `x` with shape
    /// (n_samples, n_features) and an array of target classes to predict,
    /// create a `MultiFittedLogisticRegression` object which allows making
    /// predictions. The target classes can have any number of discrete values.
    ///
    /// This method returns an error if any of the preconditions are violated,
    /// i.e. any values are `Inf` or `NaN`, `y` doesn't have as many items as
    /// `x` has rows, or if other parameters (gradient_tolerance, alpha) have
    /// been set to inalid values. The input features are also strongly recommended to be
    /// normalized to ensure numerical stability.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let (x, y) = (dataset.records(), dataset.targets());
        let (classes, target) = label_classes_multi(y)?;
        self.validate_data(x, &target)?;
        let problem = self.setup_problem(x, target);
        let solver = self.setup_solver();
        let init_params = self.setup_init_params((x.ncols(), classes.len()));
        let result = self.run_solver(problem, solver, init_params)?;

        let params = result
            .state
            .best_param
            .unwrap_or(self.setup_init_params((x.ncols(), classes.len())));
        let (w, intercept) = convert_params(x.ncols(), params.as_array());
        Ok(MultiFittedLogisticRegression::new(
            intercept.to_owned(),
            w.to_owned(),
            classes,
        ))
    }
}

/// Identify the distinct values of the classes `y` and associate
/// the target labels `-1.0` and `1.0` to it. -1.0 always labels the
/// smaller class (by PartialOrd) and 1.0 always labels the larger
/// class.
///
/// It is an error to have more than two classes.
fn label_classes<F, T, C>(y: T) -> Result<(BinaryClassLabels<F, C>, Array1<F>)>
where
    F: Float,
    T: AsSingleTargets<Elem = C>,
    C: Ord + Clone,
{
    let y = y.as_single_targets();

    let mut binary_classes = [None, None];
    for class in y {
        binary_classes = match binary_classes {
            [None, None] => [Some((class, 1)), None],
            [Some((c, count)), c2] if c == class => [Some((class, count + 1)), c2],
            [c1, Some((c, count))] if c == class => [c1, Some((class, count + 1))],
            [Some(c1), None] => [Some(c1), Some((class, 1))],
            [None, Some(_)] => unreachable!("impossible binary class array"),
            [Some(_), Some(_)] => return Err(Error::TooManyClasses),
        };
    }

    let (class_a, class_b) = match binary_classes {
        [Some(a), Some(b)] => (a, b),
        _ => return Err(Error::TooFewClasses),
    };

    // Sort by label value (Ord), not by encounter order or count.
    // The smaller label is always negative (-1),
    // the larger label is always positive (+1).
    let (neg_class, pos_class) = if class_a.0 < class_b.0 {
        (class_a, class_b)
    } else {
        (class_b, class_a)
    };

    let target_array = y
        .into_iter()
        .map(|x| {
            if x == pos_class.0 {
                F::POSITIVE_LABEL
            } else {
                F::NEGATIVE_LABEL
            }
        })
        .collect::<Array1<_>>();

    Ok((
        BinaryClassLabels {
            pos: ClassLabel {
                class: pos_class.0.clone(),
                label: F::POSITIVE_LABEL,
            },
            neg: ClassLabel {
                class: neg_class.0.clone(),
                label: F::NEGATIVE_LABEL,
            },
        },
        target_array,
    ))
}

/// Identify the distinct values of the classes in `y` and map each value to an integer. Smaller
/// classes (by `PartialOrd`) map to smaller integers. Returns the mapping along with a one-hot
/// encoding of the numerical labels corresponding to `y`.
fn label_classes_multi<F, T, C>(y: T) -> Result<(Vec<C>, Array2<F>)>
where
    F: Float,
    T: AsSingleTargets<Elem = C>,
    C: Ord + Clone,
{
    let y_single_target = y.as_single_targets();
    let mut classes = y_single_target.to_vec();
    // Dedup the list of classes
    classes.sort();
    classes.dedup();

    let mut onehot = Array2::zeros((y_single_target.len(), classes.len()));
    Zip::from(onehot.rows_mut())
        .and(&y_single_target)
        .for_each(|mut oh_row, cls| {
            let idx = classes.binary_search(cls).unwrap();
            oh_row[idx] = F::one();
        });
    Ok((classes, onehot))
}

/// Conditionally split the feature vector `w` into parameter vector and
/// intercept parameter.
/// Dimensions of `w` are either (f) or (f, n_classes)
fn convert_params<F: Float, D: Dimension + RemoveAxis>(
    n_features: usize,
    w: &Array<F, D>,
) -> (ArrayView<'_, F, D>, CowArray<'_, F, D::Smaller>) {
    let nrows = w.shape()[0];
    if n_features == nrows {
        (
            w.view(),
            Array::zeros(w.raw_dim().remove_axis(Axis(0))).into(),
        )
    } else if n_features + 1 == nrows {
        (
            w.slice_axis(Axis(0), Slice::from(..n_features)),
            w.index_axis(Axis(0), n_features).into(),
        )
    } else {
        panic!(
            "Unexpected length of parameter vector `w`, exected {} or {}, found {}",
            n_features,
            n_features + 1,
            nrows
        );
    }
}

/// Replaces every element of `x` by its exponential.
fn exp_in_place<F: linfa::Float>(x: &mut [F]) {
    for v in x {
        *v = v.exp();
    }
}

/// Replaces every element of `x` by its logistic function, `1 / (1 + exp(-x))`.
fn sigmoid_in_place<F: linfa::Float>(x: &mut [F]) {
    for v in x {
        *v = F::one() / (F::one() + (-*v).exp());
    }
}

/// Replaces every element of `x` by the log of its logistic function, in a numerically stable
/// way.
///
/// Taken from scikit-learn
/// https://github.com/scikit-learn/scikit-learn/blob/0.23.1/sklearn/utils/_logistic_sigmoid.pyx
///
/// See the blog post describing this implementation:
/// http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
fn log_sigmoid_in_place<F: linfa::Float>(x: &mut [F]) {
    for v in x {
        *v = if *v > F::zero() {
            -(F::one() + (-*v).exp()).ln()
        } else {
            *v - (F::one() + v.exp()).ln()
        };
    }
}

/// The elements of an owned array, in memory order.
fn elements_mut<F, D: Dimension>(a: &mut Array<F, D>) -> &mut [F] {
    a.as_slice_memory_order_mut()
        .expect("arrays built by arithmetic or `dot` are contiguous")
}

/// `xᵀ · v`, computed as a matrix product.
///
/// Without BLAS, ndarray computes a matrix-vector product one row of `xᵀ` at a time, which walks
/// each column of a row-major `x` with a stride of a full row: one cache line per element once
/// `x` has more than a few columns. The matrix product packs `x` into cache-sized blocks instead.
fn xt_dot<F: Float, A: Data<Elem = F>>(x: &ArrayBase<A, Ix2>, v: &Array1<F>) -> Array1<F> {
    let v = v.view().insert_axis(Axis(1));
    x.t().dot(&v).remove_axis(Axis(1))
}

/// The margins `y z` of the binary logistic loss, where `z = x params + intercept (+ offset)`.
fn binary_margins<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array1<F>,
    w: &Array1<F>,
    offset: Option<&Array1<F>>,
) -> Array1<F> {
    let n_features = x.shape()[1];
    let (params, intercept) = convert_params(n_features, w);
    let yz = x.dot(&params.into_shape_with_order((params.len(), 1)).unwrap()) + intercept;
    let len = yz.len();
    let mut yz = yz.into_shape_with_order(len).unwrap();

    if let Some(off) = offset {
        yz += off;
    }

    yz *= y;
    yz
}

/// Computes the logistic loss assuming the training labels $y \in {-1, 1}$, from the margins
/// `yz` of [`binary_margins`].
///
/// Because the logistic function fullfills $\sigma(-z) = 1 - \sigma(z)$
/// we can write $P(y=1|z) = \sigma(z) = \sigma(yz)$ and
/// $P(y=-1|z) = 1 - P(y=1|z) = 1 - \sigma(z) = \sigma(-z) = \sigma(yz)$, so
/// $P(y|z) = \sigma(yz)$ for both $y=1$ and $y=-1$.
///
/// Thus, the log loss can be written as
/// $$-\sum_{i=1}^{N} \log(\sigma(y_i z_i)) + \frac{\alpha}{2}\text{params}^T\text{params}$$
fn logistic_loss_from_margins<F: Float>(
    n_features: usize,
    yz: &Array1<F>,
    alpha: F,
    w: &Array1<F>,
) -> F {
    let (params, _) = convert_params(n_features, w);
    let mut log_sigmoid = yz.clone();
    log_sigmoid_in_place(elements_mut(&mut log_sigmoid));
    -log_sigmoid.sum() + F::cast(0.5) * alpha * params.dot(&params)
}

/// Computes the gradient of the logistic loss from the margins `yz` of [`binary_margins`].
fn logistic_grad_from_margins<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array1<F>,
    yz: Array1<F>,
    alpha: F,
    w: &Array1<F>,
) -> Array1<F> {
    let n_features = x.shape()[1];
    let (params, _) = convert_params(n_features, w);
    // `(σ(yz) - 1) y`, the derivative of the loss with respect to `z`.
    let mut r = yz;
    sigmoid_in_place(elements_mut(&mut r));
    r -= F::one();
    r *= y;
    let dw = xt_dot(x, &r) + (&params * alpha);
    if w.len() == n_features + 1 {
        let mut grad = Array::zeros(w.len());
        grad.slice_mut(s![..n_features]).assign(&dw);
        grad[n_features] = r.sum();
        grad
    } else {
        dw
    }
}

/// Computes the logistic loss assuming the training labels $y \in {-1, 1}$.
#[cfg(test)]
fn logistic_loss<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array1<F>,
    alpha: F,
    w: &Array1<F>,
    offset: Option<&Array1<F>>,
) -> F {
    let yz = binary_margins(x, y, w, offset);
    logistic_loss_from_margins(x.shape()[1], &yz, alpha, w)
}

/// Computes the gradient of the logistic loss function
#[cfg(test)]
fn logistic_grad<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array1<F>,
    alpha: F,
    w: &Array1<F>,
    offset: Option<&Array1<F>>,
) -> Array1<F> {
    let yz = binary_margins(x, y, w, offset);
    logistic_grad_from_margins(x, y, yz, alpha, w)
}

/// `log(softmax(H))` and `softmax(H)`, where H is `X . W + b`, with the same `exp` for both.
/// `X` is the input (n_samples * n_features), `W` is the params (n_features * n_classes), `b` is
/// the intercept vector (n_classes).
///
/// `log(softmax(H))` is `H - log(sum(exp(H)))` per row, with the log of the sum computed in a
/// numerically stable way, `max + ln(exp(h1-max) + exp(h2-max) + ...)`. See
/// https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
fn multi_softmax<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    w: &Array2<F>, // This parameter includes `W` and `b`
) -> (Array2<F>, Array2<F>) {
    let n_features = x.shape()[1];
    let (params, intercept) = convert_params(n_features, w);
    // Compute H
    let h = x.dot(&params) + intercept;
    // Max value of the whole array
    let max = h.iter().copied().reduce(F::max).unwrap();
    let mut exp = h.mapv(|v| v - max);
    exp_in_place(elements_mut(&mut exp));
    let sum = exp
        .fold_axis(Axis(1), F::zero(), |acc, elem| *acc + *elem)
        .mapv_into(|e| e.max(F::cast(1e-15)))
        .insert_axis(Axis(1));
    let log_prob = &h - &sum.mapv(|e| e.ln() + max);
    let prob = exp / &sum;
    (log_prob, prob)
}

/// Computes loss function of `-sum(Y * log(softmax(H))) + alpha/2 * norm(W)` from `log(softmax(H))`
fn multi_logistic_loss_from_log_prob<F: Float>(
    n_features: usize,
    log_prob: &Array2<F>,
    y: &Array2<F>,
    alpha: F,
    w: &Array2<F>,
) -> F {
    let (params, _) = convert_params(n_features, w);
    -elem_dot(log_prob, y) + F::cast(0.5) * alpha * elem_dot(&params, &params)
}

/// Computes multinomial gradients for `W` and `b` from `softmax(H)` and combine them.
/// Gradient for `W` is `Xt . (softmax(H) - Y) + alpha * W`.
/// Gradient for `b` is `sum(softmax(H) - Y)`.
fn multi_logistic_grad_from_prob<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    prob: Array2<F>,
    y: &Array2<F>,
    alpha: F,
    w: &Array2<F>,
) -> Array2<F> {
    let n_features = x.shape()[1];
    let (params, _) = convert_params(n_features, w);
    let n_classes = params.ncols();
    let intercept = w.nrows() > n_features;
    let mut grad = Array::zeros((n_features + intercept as usize, n_classes));

    let diff = prob - y;
    // Compute gradient for `W` and place it at start of the grad matrix
    let dw = x.t().dot(&diff) + (&params * alpha);
    grad.slice_mut(s![..n_features, ..]).assign(&dw);
    // Compute gradient for `b` and place it at end of grad matrix
    if intercept {
        grad.row_mut(n_features).assign(&diff.sum_axis(Axis(0)));
    }
    grad
}

/// Computes loss function of `-sum(Y * log(softmax(H))) + alpha/2 * norm(W)`
#[cfg(test)]
fn multi_logistic_loss<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array2<F>,
    alpha: F,
    w: &Array2<F>,
) -> F {
    let (log_prob, _) = multi_softmax(x, w);
    multi_logistic_loss_from_log_prob(x.shape()[1], &log_prob, y, alpha, w)
}

/// Computes multinomial gradients for `W` and `b` and combine them.
#[cfg(test)]
fn multi_logistic_grad<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array2<F>,
    alpha: F,
    w: &Array2<F>,
) -> Array2<F> {
    let (_, prob) = multi_softmax(x, w);
    multi_logistic_grad_from_prob(x, prob, y, alpha, w)
}

/// Replaces every row of `m` by its softmax, `exp(n - max) / sum(exp(n - max))` with the row's
/// max, a numerically stable version of softmax.
fn softmax_rows<F: linfa::Float>(m: &mut Array2<F>) {
    for mut row in m.rows_mut() {
        let max = row.iter().copied().reduce(F::max).unwrap();
        row.mapv_inplace(|n| n - max);
    }
    exp_in_place(elements_mut(m));
    for mut row in m.rows_mut() {
        let sum = row.sum();
        row.mapv_inplace(|n| n / sum);
    }
}

/// A fitted logistic regression which can make predictions
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct FittedLogisticRegression<F: Float, C: PartialOrd + Clone> {
    threshold: F,
    intercept: F,
    params: Array1<F>,
    labels: BinaryClassLabels<F, C>,
}

impl<F: Float, C: PartialOrd + Clone> FittedLogisticRegression<F, C> {
    fn new(
        intercept: F,
        params: Array1<F>,
        labels: BinaryClassLabels<F, C>,
    ) -> FittedLogisticRegression<F, C> {
        FittedLogisticRegression {
            threshold: F::cast(0.5),
            intercept,
            params,
            labels,
        }
    }

    /// Set the probability threshold for which the 'positive' class will be
    /// predicted. Defaults to 0.5.
    pub fn set_threshold(mut self, threshold: F) -> FittedLogisticRegression<F, C> {
        if threshold < F::zero() || threshold > F::one() {
            panic!("FittedLogisticRegression::set_threshold: threshold needs to be between 0.0 and 1.0");
        }
        self.threshold = threshold;
        self
    }

    pub fn intercept(&self) -> F {
        self.intercept
    }

    pub fn params(&self) -> &Array1<F> {
        &self.params
    }

    /// Get the model positive and negative classes mapped to their
    /// corresponding problem input labels.
    pub fn labels(&self) -> &BinaryClassLabels<F, C> {
        &self.labels
    }

    /// Given a feature matrix, predict the probabilities that a sample
    /// should be classified as the larger of the two classes learned when the
    /// model was fitted.
    pub fn predict_probabilities<A: Data<Elem = F>>(&self, x: &ArrayBase<A, Ix2>) -> Array1<F> {
        let mut probs = x.dot(&self.params) + self.intercept;
        sigmoid_in_place(elements_mut(&mut probs));
        probs
    }
}

impl<C: PartialOrd + Clone + Default, F: Float, D: Data<Elem = F>>
    PredictInplace<ArrayBase<D, Ix2>, Array1<C>> for FittedLogisticRegression<F, C>
{
    /// Given a feature matrix, predict the classes learned when the model was
    /// fitted.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<C>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );
        assert_eq!(
            x.ncols(),
            self.params.len(),
            "Number of data features must match the number of features the model was trained with."
        );

        let pos_class = &self.labels.pos.class;
        let neg_class = &self.labels.neg.class;
        Zip::from(&self.predict_probabilities(x))
            .and(y)
            .for_each(|prob, out| {
                *out = if *prob >= self.threshold {
                    pos_class.clone()
                } else {
                    neg_class.clone()
                }
            });
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<C> {
        Array1::default(x.nrows())
    }
}

/// A fitted multinomial logistic regression which can make predictions
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct MultiFittedLogisticRegression<F, C: PartialOrd + Clone> {
    intercept: Array1<F>,
    params: Array2<F>,
    classes: Vec<C>,
}

impl<F: Float, C: PartialOrd + Clone> MultiFittedLogisticRegression<F, C> {
    fn new(intercept: Array1<F>, params: Array2<F>, classes: Vec<C>) -> Self {
        Self {
            intercept,
            params,
            classes,
        }
    }

    pub fn intercept(&self) -> &Array1<F> {
        &self.intercept
    }

    pub fn params(&self) -> &Array2<F> {
        &self.params
    }

    /// Return non-normalized probabilities (n_samples * n_classes)
    fn predict_nonorm_probabilities<A: Data<Elem = F>>(&self, x: &ArrayBase<A, Ix2>) -> Array2<F> {
        x.dot(&self.params) + &self.intercept
    }

    /// Return normalized probabilities for each output class. The output dimensions are (n_samples
    /// * n_classes).
    pub fn predict_probabilities<A: Data<Elem = F>>(&self, x: &ArrayBase<A, Ix2>) -> Array2<F> {
        let mut probs = self.predict_nonorm_probabilities(x);
        softmax_rows(&mut probs);
        probs
    }

    /// Get the list of class labels, which maps the numerical class indices to the labels
    pub fn classes(&self) -> &[C] {
        &self.classes
    }
}

impl<C: PartialOrd + Clone + Default, F: Float, D: Data<Elem = F>>
    PredictInplace<ArrayBase<D, Ix2>, Array1<C>> for MultiFittedLogisticRegression<F, C>
{
    /// Given a feature matrix, predict the classes learned when the model was
    /// fitted.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<C>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );
        assert_eq!(
            x.ncols(),
            self.params.nrows(),
            "Number of data features must match the number of features the model was trained with."
        );

        let probs = self.predict_nonorm_probabilities(x);
        Zip::from(probs.rows()).and(y).for_each(|prob_row, out| {
            let idx = prob_row.argmax().unwrap();
            *out = self.classes[idx].clone();
        });
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<C> {
        Array1::default(x.nrows())
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct ClassLabel<F, C: PartialOrd> {
    pub class: C,
    pub label: F,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct BinaryClassLabels<F, C: PartialOrd> {
    pub pos: ClassLabel<F, C>,
    pub neg: ClassLabel<F, C>,
}

/// Internal representation of a logistic regression problem.
/// This data structure exists to be handed to Argmin.
struct LogisticRegressionProblem<'a, F: Float, A: Data<Elem = F>, D: Dimension> {
    x: &'a ArrayBase<A, Ix2>,
    target: Array<F, D>,
    alpha: F,
    offset: Option<Array1<F>>,
    /// The parameters of the last cost evaluation, with what the gradient at the same parameters
    /// reuses: the margins `y z` (binary) or `softmax(H)` (multinomial). The line search evaluates
    /// the cost and then the gradient at each point it tries.
    reuse: Reuse<F, D>,
    /// The last gradient and its parameters: LBFGS asks again for the gradient at the point where
    /// the line search ended, which the line search has just computed.
    last_gradient: Reuse<F, D>,
}

impl<'a, F: Float, A: Data<Elem = F>, D: Dimension> LogisticRegressionProblem<'a, F, A, D> {
    fn new(
        x: &'a ArrayBase<A, Ix2>,
        target: Array<F, D>,
        alpha: F,
        offset: Option<Array1<F>>,
    ) -> Self {
        LogisticRegressionProblem {
            x,
            target,
            alpha,
            offset,
            reuse: RefCell::new(None),
            last_gradient: RefCell::new(None),
        }
    }

    /// What the last cost evaluation left for the gradient, if it was at `w`.
    fn take_reusable(&self, w: &Array<F, D>) -> Option<Array<F, D>> {
        match self.reuse.borrow_mut().take() {
            Some((at, reusable)) if at == *w => Some(reusable),
            _ => None,
        }
    }

    /// The last gradient, if it was at `w`.
    fn last_gradient(&self, w: &Array<F, D>) -> Option<Array<F, D>> {
        match &*self.last_gradient.borrow() {
            Some((at, grad)) if at == w => Some(grad.clone()),
            _ => None,
        }
    }

    /// Records the gradient at `w`.
    fn set_last_gradient(&self, w: &Array<F, D>, grad: &Array<F, D>) {
        *self.last_gradient.borrow_mut() = Some((w.clone(), grad.clone()));
    }
}

/// Parameters, and what the gradient at those parameters reuses from the cost.
type Reuse<F, D> = RefCell<Option<(Array<F, D>, Array<F, D>)>>;

type LogisticRegressionProblem1<'a, F, A> = LogisticRegressionProblem<'a, F, A, Ix1>;
type LogisticRegressionProblem2<'a, F, A> = LogisticRegressionProblem<'a, F, A, Ix2>;

impl<F: Float, A: Data<Elem = F>> CostFunction for LogisticRegressionProblem<'_, F, A, Ix1> {
    type Param = ArgminParam<F, Ix1>;
    type Output = F;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let w = p.as_array();
        let yz = binary_margins(self.x, &self.target, w, self.offset.as_ref());
        let cost = logistic_loss_from_margins(self.x.shape()[1], &yz, self.alpha, w);
        *self.reuse.borrow_mut() = Some((w.clone(), yz));
        Ok(cost)
    }
}

impl<F: Float, A: Data<Elem = F>> Gradient for LogisticRegressionProblem<'_, F, A, Ix1> {
    type Param = ArgminParam<F, Ix1>;
    type Gradient = ArgminParam<F, Ix1>;

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> std::result::Result<Self::Param, argmin::core::Error> {
        let w = p.as_array();
        if let Some(grad) = self.last_gradient(w) {
            return Ok(ArgminParam(grad));
        }
        let yz = self
            .take_reusable(w)
            .unwrap_or_else(|| binary_margins(self.x, &self.target, w, self.offset.as_ref()));
        let grad = logistic_grad_from_margins(self.x, &self.target, yz, self.alpha, w);
        self.set_last_gradient(w, &grad);
        Ok(ArgminParam(grad))
    }
}

impl<F: Float, A: Data<Elem = F>> CostFunction for LogisticRegressionProblem<'_, F, A, Ix2> {
    type Param = ArgminParam<F, Ix2>;
    type Output = F;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let w = p.as_array();
        let (log_prob, prob) = multi_softmax(self.x, w);
        let cost = multi_logistic_loss_from_log_prob(
            self.x.shape()[1],
            &log_prob,
            &self.target,
            self.alpha,
            w,
        );
        *self.reuse.borrow_mut() = Some((w.clone(), prob));
        Ok(cost)
    }
}

impl<F: Float, A: Data<Elem = F>> Gradient for LogisticRegressionProblem<'_, F, A, Ix2> {
    type Param = ArgminParam<F, Ix2>;
    type Gradient = ArgminParam<F, Ix2>;

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> std::result::Result<Self::Param, argmin::core::Error> {
        let w = p.as_array();
        if let Some(grad) = self.last_gradient(w) {
            return Ok(ArgminParam(grad));
        }
        let prob = self
            .take_reusable(w)
            .unwrap_or_else(|| multi_softmax(self.x, w).1);
        let grad = multi_logistic_grad_from_prob(self.x, prob, &self.target, self.alpha, w);
        self.set_last_gradient(w, &grad);
        Ok(ArgminParam(grad))
    }
}

trait SolvableProblem<F: Float, D: Dimension>: Gradient + Sized {
    type Solver: Solver<Self, IterStateType<F, D>>;
}

impl<F: Float, A: Data<Elem = F>> SolvableProblem<F, Ix1> for LogisticRegressionProblem1<'_, F, A> {
    type Solver = LBFGSType1<F>;
}

impl<F: Float, A: Data<Elem = F>> SolvableProblem<F, Ix2> for LogisticRegressionProblem2<'_, F, A> {
    type Solver = LBFGSType2<F>;
}

#[cfg(test)]
mod test {
    extern crate linfa;

    use super::Error;
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq, AbsDiffEq};
    use linfa::prelude::*;
    use ndarray::{array, Array2, Dim, Ix};

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<LogisticRegressionParams<f64, Dim<[Ix; 0]>>>();
        has_autotraits::<LogisticRegressionValidParams<f64, Dim<[Ix; 0]>>>();
        has_autotraits::<ArgminParam<f64, Dim<[Ix; 0]>>>();
    }

    /// Test that the logistic loss function works as expected.
    /// The expected values were obtained from running sklearn's
    /// _logistic_loss_and_grad function.
    #[test]
    fn test_logistic_loss() {
        let x = array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ws = [
            array![0.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 0.0],
            array![1.0, 1.0],
            array![0.0, -1.0],
            array![-1.0, 0.0],
            array![-1.0, -1.0],
        ];
        let alphas = &[0.0, 1.0, 10.0];
        let expecteds = vec![
            6.931471805599453,
            6.931471805599453,
            6.931471805599453,
            4.652158847349118,
            4.652158847349118,
            4.652158847349118,
            2.8012999588008323,
            3.3012999588008323,
            7.801299958800833,
            2.783195429782239,
            3.283195429782239,
            7.783195429782239,
            10.652158847349117,
            10.652158847349117,
            10.652158847349117,
            41.80129995880083,
            42.30129995880083,
            46.80129995880083,
            47.78319542978224,
            48.28319542978224,
            52.78319542978224,
        ];

        for ((w, alpha), exp) in ws
            .iter()
            .flat_map(|w| alphas.iter().map(move |&alpha| (w, alpha)))
            .zip(&expecteds)
        {
            assert_abs_diff_eq!(logistic_loss(&x, &y, alpha, w, None), *exp);
        }
    }

    /// Test that the logistic grad function works as expected.
    /// The expected values were obtained from running sklearn's
    /// _logistic_loss_and_grad function.
    #[test]
    fn test_logistic_grad() {
        let x = array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ws = [
            array![0.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 0.0],
            array![1.0, 1.0],
            array![0.0, -1.0],
            array![-1.0, 0.0],
            array![-1.0, -1.0],
        ];
        let alphas = &[0.0, 1.0, 10.0];
        let expecteds = vec![
            array![-19.5, -3.],
            array![-19.5, -3.],
            array![-19.5, -3.],
            array![-10.48871543, -1.61364853],
            array![-10.48871543, -1.61364853],
            array![-10.48871543, -1.61364853],
            array![-0.13041554, -0.02852148],
            array![0.86958446, -0.02852148],
            array![9.86958446, -0.02852148],
            array![-0.04834401, -0.01058067],
            array![0.95165599, -0.01058067],
            array![9.95165599, -0.01058067],
            array![-28.51128457, -4.38635147],
            array![-28.51128457, -4.38635147],
            array![-28.51128457, -4.38635147],
            array![-38.86958446, -5.97147852],
            array![-39.86958446, -5.97147852],
            array![-48.86958446, -5.97147852],
            array![-38.95165599, -5.98941933],
            array![-39.95165599, -5.98941933],
            array![-48.95165599, -5.98941933],
        ];

        for ((w, alpha), exp) in ws
            .iter()
            .flat_map(|w| alphas.iter().map(move |&alpha| (w, alpha)))
            .zip(&expecteds)
        {
            let actual = logistic_grad(&x, &y, alpha, w, None);
            assert!(actual.abs_diff_eq(exp, 1e-8));
        }
    }

    #[test]
    fn simple_example_1() {
        let log_reg = LogisticRegression::default();
        let x = array![[-1.0], [-0.01], [0.01], [1.0]];
        let y = array![0, 0, 1, 1];
        let dataset = Dataset::new(x, y);
        let res = log_reg.fit(&dataset).unwrap();
        assert_abs_diff_eq!(res.intercept(), 0.0);
        assert!(res.params().abs_diff_eq(&array![0.681], 1e-3));
        assert_eq!(
            &res.predict(dataset.records()),
            dataset.targets().as_single_targets()
        );
    }

    #[test]
    fn simple_example_1_cats_dogs() {
        let log_reg = LogisticRegression::default();
        let x = array![[0.01], [1.0], [-1.0], [-0.01]];
        let y = array!["dog", "dog", "cat", "cat"];
        let dataset = Dataset::new(x, y);
        let res = log_reg.fit(&dataset).unwrap();
        assert_abs_diff_eq!(res.intercept(), 0.0);
        assert!(res.params().abs_diff_eq(&array![0.681], 1e-3));
        assert!(res
            .predict_probabilities(dataset.records())
            .abs_diff_eq(&array![0.501, 0.664, 0.335, 0.498], 1e-3));
        assert_eq!(
            &res.predict(dataset.records()),
            dataset.targets().as_single_targets()
        );
        assert_eq!(res.labels().pos.class, "dog");
        assert_eq!(res.labels().neg.class, "cat");
    }

    #[test]
    fn simple_example_2() {
        let log_reg = LogisticRegression::default().alpha(1.0);
        let x = array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let dataset = Dataset::new(x, y);
        let res = log_reg.fit(&dataset).unwrap();
        assert_eq!(
            &res.predict(dataset.records()),
            dataset.targets().as_single_targets()
        );
    }

    #[test]
    fn simple_example_3() {
        let x = array![[1.0], [0.0], [1.0], [0.0]];
        let y = array![1, 0, 1, 0];
        let dataset = DatasetBase::new(x, y);
        let model = LogisticRegression::default().fit(&dataset).unwrap();

        let pred = model.predict(&dataset.records);
        assert_eq!(dataset.targets(), pred);
    }

    #[test]
    fn rejects_mismatching_x_y() {
        let log_reg = LogisticRegression::default();
        let x = array![[-1.0], [-0.01], [0.01]];
        let y = array![0, 0, 1, 1];
        let res = log_reg.fit(&Dataset::new(x, y));
        assert!(matches!(res.unwrap_err(), Error::MismatchedShapes(3, 4)));
    }

    #[test]
    fn rejects_inf_values() {
        let infs = &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        let inf_xs: Vec<_> = infs.iter().map(|&inf| array![[1.0], [inf]]).collect();
        let log_reg = LogisticRegression::default();
        let normal_x = array![[-1.0], [1.0]];
        let y = array![0, 1];
        for inf_x in &inf_xs {
            let res = log_reg.fit(&DatasetBase::new(inf_x.view(), &y));
            assert!(matches!(res.unwrap_err(), Error::InvalidValues));
        }
        for inf in infs {
            let log_reg = LogisticRegression::default().alpha(*inf);
            let res = log_reg.fit(&DatasetBase::new(normal_x.view(), &y));
            assert!(matches!(res.unwrap_err(), Error::InvalidAlpha));
        }
        let mut non_positives = infs.to_vec();
        non_positives.push(-1.0);
        non_positives.push(0.0);
        for inf in &non_positives {
            let log_reg = LogisticRegression::default().gradient_tolerance(*inf);
            let res = log_reg.fit(&Dataset::new(normal_x.to_owned(), y.to_owned()));
            assert!(matches!(res.unwrap_err(), Error::InvalidGradientTolerance));
        }
    }

    #[test]
    fn validates_initial_params() {
        let infs = &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        let normal_x = array![[-1.0], [1.0]];
        let normal_y = array![0, 1];
        let dataset = Dataset::new(normal_x, normal_y);
        for inf in infs {
            let log_reg = LogisticRegression::default().initial_params(array![*inf, 0.0]);
            let res = log_reg.fit(&dataset);
            assert!(matches!(res.unwrap_err(), Error::InvalidInitialParameters));
        }
        {
            let log_reg = LogisticRegression::default().initial_params(array![0.0, 0.0, 0.0]);
            let res = log_reg.fit(&dataset);
            assert!(matches!(
                res.unwrap_err(),
                Error::InitialParameterFeaturesMismatch {
                    rows: 3,
                    n_features: 2
                }
            ));
        }
        {
            let log_reg = LogisticRegression::default()
                .with_intercept(false)
                .initial_params(array![0.0, 0.0]);
            let res = log_reg.fit(&dataset);
            assert!(matches!(
                res.unwrap_err(),
                Error::InitialParameterFeaturesMismatch {
                    rows: 2,
                    n_features: 1
                }
            ));
        }
    }

    #[test]
    fn uses_initial_params() {
        let params = array![1.2, -4.12];
        let log_reg = LogisticRegression::default()
            .initial_params(params)
            .max_iterations(5);
        let x = array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let dataset = Dataset::new(x, y);
        let res = log_reg.fit(&dataset).unwrap();
        assert!(res.intercept().abs_diff_eq(&-4.124, 1e-3));
        assert!(res.params().abs_diff_eq(&array![1.181], 1e-3));
        assert_eq!(
            &res.predict(dataset.records()),
            dataset.targets().as_single_targets()
        );

        // Test serialization
        #[cfg(feature = "serde")]
        {
            let ser = rmp_serde::to_vec(&res).unwrap();
            let unser: FittedLogisticRegression<f32, f32> = rmp_serde::from_slice(&ser).unwrap();

            let x = array![[1.0]];
            let y_hat = unser.predict(&x);

            assert!(y_hat[0] == 0.0);
        }
    }

    #[test]
    fn works_with_f32() {
        let log_reg = LogisticRegression::default();
        let x: Array2<f32> = array![[-1.0], [-0.01], [0.01], [1.0]];
        let y = array![0, 0, 1, 1];
        let dataset = Dataset::new(x, y);
        let res = log_reg.fit(&dataset).unwrap();
        assert_abs_diff_eq!(res.intercept(), 0.0_f32);
        assert!(res.params().abs_diff_eq(&array![0.682_f32], 1e-3));
        assert_eq!(
            &res.predict(dataset.records()),
            dataset.targets().as_single_targets()
        );
    }

    /// The gradient reuses what the cost computed, and the last gradient, only at the same
    /// parameters: in any order of calls, every gradient equals one computed from scratch.
    #[test]
    fn test_problem_reuses_only_at_same_params() {
        let x = array![
            [0.5, -1.0],
            [1.5, 0.2],
            [-0.3, 0.8],
            [2.0, -0.7],
            [-1.2, -0.4]
        ];
        let w1 = array![0.3, -0.2, 0.1];
        let w2 = array![-0.4, 0.6, -0.2];

        let y = array![1.0, -1.0, 1.0, 1.0, -1.0];
        let fresh = |w: &Array1<f64>| logistic_grad(&x, &y, 0.5, w, None);
        let problem = LogisticRegressionProblem::new(&x, y.clone(), 0.5, None);
        let (p1, p2) = (ArgminParam(w1.clone()), ArgminParam(w2.clone()));
        assert_eq!(
            problem.cost(&p1).unwrap(),
            logistic_loss(&x, &y, 0.5, &w1, None)
        );
        assert_eq!(problem.gradient(&p2).unwrap().0, fresh(&w2));
        assert_eq!(problem.gradient(&p1).unwrap().0, fresh(&w1));
        problem.cost(&p2).unwrap();
        assert_eq!(problem.gradient(&p2).unwrap().0, fresh(&w2));
        assert_eq!(problem.gradient(&p2).unwrap().0, fresh(&w2));
        assert_eq!(problem.gradient(&p1).unwrap().0, fresh(&w1));

        let y = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ];
        let w1 = array![[0.3, -0.2, 0.1], [0.5, 0.0, -0.5], [0.1, 0.2, 0.3]];
        let w2 = &w1 * -2.0;
        let fresh = |w: &Array2<f64>| multi_logistic_grad(&x, &y, 0.5, w);
        let problem = LogisticRegressionProblem::new(&x, y.clone(), 0.5, None);
        let (p1, p2) = (ArgminParam(w1.clone()), ArgminParam(w2.clone()));
        assert_eq!(
            problem.cost(&p1).unwrap(),
            multi_logistic_loss(&x, &y, 0.5, &w1)
        );
        assert_eq!(problem.gradient(&p2).unwrap().0, fresh(&w2));
        assert_eq!(problem.gradient(&p1).unwrap().0, fresh(&w1));
        problem.cost(&p2).unwrap();
        assert_eq!(problem.gradient(&p2).unwrap().0, fresh(&w2));
        assert_eq!(problem.gradient(&p2).unwrap().0, fresh(&w2));
        assert_eq!(problem.gradient(&p1).unwrap().0, fresh(&w1));
    }

    #[test]
    fn test_log_sum_exp() {
        let data = array![[3.3, 0.4, -2.1], [0.4, 2.2, -0.1], [1., 0., -1.]];
        // With `X = I` and no intercept, `H` is `data` and `H - log(softmax(H))` its log-sum-exp.
        let (log_prob, _) = multi_softmax(&Array2::eye(3), &data);
        let out = (&data - &log_prob).column(0).to_owned();
        assert_abs_diff_eq!(out, array![3.35783, 2.43551, 1.40761], epsilon = 1e-5);
    }

    #[test]
    fn test_softmax() {
        let mut data = array![[3.3, 5.5, 0.1, -4.4, 8.0]];
        softmax_rows(&mut data);
        let data = data.row(0);
        assert_relative_eq!(
            data,
            array![0.0083324, 0.075200047, 0.000339647, 0.000003773, 0.91612413],
            epsilon = 1e-8
        );
        assert_abs_diff_eq!(data.sum(), 1.0);
    }

    #[test]
    fn test_multi_logistic_loss_grad() {
        let x = array![
            [0.0, 0.5],
            [1.0, -1.0],
            [2.0, -2.0],
            [3.0, -3.0],
            [4.0, -4.0],
            [5.0, -5.0],
            [6.0, -6.0],
            [7.0, -7.0],
        ];
        let y = array![
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let params1 = array![[4.4, -1.2, 3.3], [3.4, 0.1, 0.0]];
        let params2 = array![[0.001, -3.2, 2.9], [0.1, 4.5, 5.7], [4.5, 2.2, 1.7]];
        let alpha = 0.6;

        {
            let (log_prob, _) = multi_softmax(&x, &params1);
            let (w, _) = convert_params(x.ncols(), &params1);
            assert_abs_diff_eq!(
                log_prob,
                array![
                    [-3.18259845e-01, -1.96825985e+00, -2.01825985e+00],
                    [-2.40463987e+00, -4.70463987e+00, -1.04639868e-01],
                    [-4.61010168e+00, -9.21010168e+00, -1.01016809e-02],
                    [-6.90100829e+00, -1.38010083e+01, -1.00829256e-03],
                    [-9.20010104e+00, -1.84001010e+01, -1.01044506e-04],
                    [-1.15000101e+01, -2.30000101e+01, -1.01301449e-05],
                    [-1.38000010e+01, -2.76000010e+01, -1.01563199e-06],
                    [-1.61000001e+01, -3.22000001e+01, -1.01826043e-07],
                ],
                epsilon = 1e-6
            );
            assert_abs_diff_eq!(w, params1);
            let loss = multi_logistic_loss(&x, &y, alpha, &params1);
            assert_abs_diff_eq!(loss, 57.11212197835295, epsilon = 1e-6);
            let grad = multi_logistic_grad(&x, &y, alpha, &params1);
            assert_abs_diff_eq!(
                grad,
                array![
                    [1.7536815, -9.71074369, 11.85706219],
                    [2.79002537, 9.12059357, -9.81061893]
                ],
                epsilon = 1e-6
            );
        }

        {
            let (log_prob, _) = multi_softmax(&x, &params2);
            let (w, _) = convert_params(x.ncols(), &params2);
            assert_abs_diff_eq!(
                log_prob,
                array![
                    [-1.06637742e+00, -1.16637742e+00, -1.06637742e+00],
                    [-4.12429463e-03, -9.90512429e+00, -5.50512429e+00],
                    [-2.74092305e-04, -1.75022741e+01, -8.20227409e+00],
                    [-1.84027855e-05, -2.51030184e+01, -1.09030184e+01],
                    [-1.23554225e-06, -3.27040012e+01, -1.36040012e+01],
                    [-8.29523046e-08, -4.03050001e+01, -1.63050001e+01],
                    [-5.56928016e-09, -4.79060000e+01, -1.90060000e+01],
                    [-3.73912013e-10, -5.55070000e+01, -2.17070000e+01]
                ],
                epsilon = 1e-6
            );
            assert_abs_diff_eq!(w, params2.slice(s![..params2.nrows() - 1, ..]));
            let loss = multi_logistic_loss(&x, &y, alpha, &params2);
            assert_abs_diff_eq!(loss, 154.8177958366479, epsilon = 1e-6);
            let grad = multi_logistic_grad(&x, &y, alpha, &params2);
            assert_abs_diff_eq!(
                grad,
                array![
                    [26.99587549, -10.91995003, -16.25532546],
                    [-27.26314882, 11.85569669, 21.58745213],
                    [5.33984376, -2.68845675, -2.65138701]
                ],
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn simple_multi_example() {
        let x = array![[-1., 0.], [0., 1.], [1., 1.]];
        let y = array![2, 1, 0];
        let log_reg = MultiLogisticRegression::default()
            .alpha(0.1)
            .initial_params(Array::zeros((3, 3)));
        let dataset = Dataset::new(x, y);
        let res = log_reg.fit(&dataset).unwrap();
        assert_eq!(res.params().dim(), (2, 3));
        assert_eq!(res.intercept().dim(), 3);
        assert_eq!(
            &res.predict(dataset.records()),
            dataset.targets().as_single_targets()
        );
    }

    #[test]
    fn simple_multi_example_2() {
        let x = array![[1.0], [0.0], [1.0], [0.0]];
        let y = array![1, 0, 1, 0];
        let dataset = DatasetBase::new(x, y);
        let model = MultiLogisticRegression::default().fit(&dataset).unwrap();

        let pred = model.predict(&dataset.records);
        assert_eq!(dataset.targets(), pred);
    }

    #[test]
    fn simple_multi_example_text() {
        let log_reg = MultiLogisticRegression::default().alpha(0.1);
        let x = array![[0.1], [1.0], [-1.0], [-0.1]];
        let y = array!["dog", "ape", "rocket", "cat"];
        let dataset = Dataset::new(x, y);
        let res = log_reg.fit(&dataset).unwrap();
        assert_eq!(res.params().dim(), (1, 4));
        assert_eq!(res.intercept().dim(), 4);
        assert_eq!(
            &res.predict(dataset.records()),
            dataset.targets().as_single_targets()
        );
    }

    #[test]
    fn multi_on_binary_problem() {
        let log_reg = MultiLogisticRegression::default().alpha(1.0);
        let x = array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let dataset = Dataset::new(x, y);
        let res = log_reg.fit(&dataset).unwrap();
        assert_eq!(res.params().dim(), (1, 2));
        assert_eq!(res.intercept().dim(), 2);
        assert_eq!(
            &res.predict(dataset.records()),
            dataset.targets().as_single_targets()
        );
    }

    #[test]
    fn reject_num_class_mismatch() {
        let n_samples = 4;
        let n_classes = 3;
        let n_features = 1;
        let x = Array2::<f64>::zeros((n_samples, n_features));
        let y = array![0, 1, 2, 0];
        let dataset = Dataset::new(x, y);

        let log_reg = MultiLogisticRegression::default()
            .with_intercept(false)
            .initial_params(Array::zeros((n_features, n_classes - 1)));
        assert!(matches!(
            log_reg.fit(&dataset).unwrap_err(),
            Error::InitialParameterClassesMismatch {
                cols: 2,
                n_classes: 3,
            }
        ));
    }

    #[test]
    fn label_order_independent() {
        let x1 = array![[-1.0], [1.0], [-0.5], [0.5]];
        let y1 = array!["cat", "dog", "cat", "dog"];

        let x2 = array![[1.0], [-1.0], [0.5], [-0.5]];
        let y2 = array!["dog", "cat", "dog", "cat"];

        let model1 = LogisticRegression::default()
            .fit(&Dataset::new(x1, y1))
            .unwrap();
        let model2 = LogisticRegression::default()
            .fit(&Dataset::new(x2, y2))
            .unwrap();

        assert_eq!(model1.labels().pos.class, "dog");
        assert_eq!(model1.labels().neg.class, "cat");
        assert_eq!(model2.labels().pos.class, "dog");
        assert_eq!(model2.labels().neg.class, "cat");

        assert_abs_diff_eq!(model1.intercept(), model2.intercept());
        assert!(model1.params().abs_diff_eq(model2.params(), 1e-6));
    }

    #[test]
    fn rejects_mismatched_offset_length() {
        let log_reg = LogisticRegression::default().offset(array![1.0, 2.0, 3.0]);
        let x = array![[-1.0], [-0.01], [0.01], [1.0]];
        let y = array![0, 0, 1, 1];
        let res = log_reg.fit(&Dataset::new(x, y));
        assert!(matches!(
            res.unwrap_err(),
            Error::OffsetLengthMismatch {
                offset_len: 3,
                n_samples: 4,
            }
        ));
    }

    #[test]
    fn zero_offset_same_as_no_offset() {
        let x = array![[-1.0], [-0.01], [0.01], [1.0]];
        let y = array![0, 0, 1, 1];

        let model_none = LogisticRegression::default()
            .fit(&Dataset::new(x.clone(), y.clone()))
            .unwrap();

        let model_zero = LogisticRegression::default()
            .offset(array![0.0, 0.0, 0.0, 0.0])
            .fit(&Dataset::new(x, y))
            .unwrap();

        assert_abs_diff_eq!(model_none.intercept(), model_zero.intercept());
        assert!(model_none.params().abs_diff_eq(model_zero.params(), 1e-6));
    }

    #[test]
    fn offset_changes_model() {
        let x = array![[-1.0], [-0.01], [0.01], [1.0]];
        let y = array![0, 0, 1, 1];

        let model_none = LogisticRegression::default()
            .fit(&Dataset::new(x.clone(), y.clone()))
            .unwrap();

        let model_offset = LogisticRegression::default()
            .offset(array![1.0, 1.0, -1.0, -1.0])
            .fit(&Dataset::new(x, y))
            .unwrap();

        assert!(
            !model_none.params().abs_diff_eq(model_offset.params(), 1e-3),
            "Offset should change the learned parameters"
        );
    }
}
