//! # Logistic Regression
//!
//! ## The Big Picture
//!
//! `linfa-logistic` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.
//!
//! ## Current state
//! `linfa-logistic` provides a pure Rust implementation of a two class [logistic regression model](struct.LogisticRegression.html).
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
use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::lbfgs::LBFGS;
use linfa::prelude::{AsTargets, DatasetBase};
use linfa::traits::{Fit, PredictInplace};
use ndarray::{
    s, Array, Array1, Array2, ArrayBase, Axis, Data, DataMut, Dimension, IntoDimension, Ix1, Ix2,
    RemoveAxis, Slice, Zip,
};
use ndarray_stats::QuantileExt;
use std::default::Default;

mod argmin_param;
mod float;

use argmin_param::{elem_dot, ArgminParam};
use float::Float;

/// A two-class logistic regression model.
///
/// Logistic regression combines linear models with
/// the sigmoid function `sigm(x) = 1/(1+exp(-x))`
/// to learn a family of functions that map the feature space to `[0,1]`.
///
/// Logistic regression is used in binary classification
/// by interpreting the predicted value as the probability that the sample
/// has label `1`. A threshold can be set in the [fitted model](struct.FittedLogisticRegression.html) to decide the minimum
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
pub struct LogisticRegressionBase<F: Float, D: Dimension> {
    alpha: F,
    fit_intercept: bool,
    max_iterations: u64,
    gradient_tolerance: F,
    initial_params: Option<Array<F, D>>,
}

pub type LogisticRegression<F> = LogisticRegressionBase<F, Ix1>;
pub type MultiLogisticRegression<F> = LogisticRegressionBase<F, Ix2>;

impl<F: Float, D: Dimension> Default for LogisticRegressionBase<F, D> {
    fn default() -> Self {
        LogisticRegressionBase::new()
    }
}

type LBFGSType<F, D> = LBFGS<MoreThuenteLineSearch<ArgminParam<F, D>, F>, ArgminParam<F, D>, F>;
type LBFGSType1<F> = LBFGSType<F, Ix1>;
type LBFGSType2<F> = LBFGSType<F, Ix2>;

impl<F: Float, D: Dimension> LogisticRegressionBase<F, D> {
    /// Creates a new LogisticRegression with default configuration.
    pub fn new() -> Self {
        LogisticRegressionBase {
            alpha: F::cast(1.0),
            fit_intercept: true,
            max_iterations: 100,
            gradient_tolerance: F::cast(1e-4),
            initial_params: None,
        }
    }

    /// Set the normalization parameter `alpha` used for L2 normalization,
    /// defaults to `1.0`.
    pub fn alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Configure if an intercept should be fitted, defaults to `true`.
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Configure the maximum number of iterations that the solver should perform,
    /// defaults to `100`.
    pub fn max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Configure the minimum change to the gradient to continue the solver,
    /// defaults to `1e-4`.
    pub fn gradient_tolerance(mut self, gradient_tolerance: F) -> Self {
        self.gradient_tolerance = gradient_tolerance;
        self
    }

    /// Configure the initial parameters from where the optimization starts.
    /// The `params` array must have at least the same number of rows as there are columns on the
    /// feature matrix `x` passed to the `fit` method
    pub fn initial_params(mut self, params: Array<F, D>) -> Self {
        self.initial_params = Some(params);
        self
    }

    /// Create the initial parameters, either from a user supplied guess
    /// or a 1-d array of `0`s.
    fn setup_init_params(&self, dims: D::Pattern) -> Array<F, D> {
        if let Some(params) = self.initial_params.as_ref() {
            params.clone()
        } else {
            let mut dims = dims.into_dimension();
            dims.as_array_view_mut()[0] += self.fit_intercept as usize;
            Array::zeros(dims)
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
        if x.iter().any(|x| !x.is_finite())
            || y.iter().any(|y| !y.is_finite())
            || !self.alpha.is_finite()
        {
            return Err(Error::InvalidValues);
        }
        if !self.gradient_tolerance.is_finite() || self.gradient_tolerance <= F::zero() {
            return Err(Error::InvalidGradientTolerance);
        }
        self.validate_init_params(x.shape()[1], y.shape().get(1).copied())?;
        Ok(())
    }

    fn validate_init_params(&self, mut n_features: usize, n_classes: Option<usize>) -> Result<()> {
        if let Some(params) = self.initial_params.as_ref() {
            let shape = params.shape();
            n_features += self.fit_intercept as usize;
            if n_features != shape[0] || n_classes != shape.get(1).copied() {
                return Err(Error::InvalidInitialParametersGuessSize);
            }
            if params.iter().any(|p| !p.is_finite()) {
                return Err(Error::InvalidInitialParametersGuess);
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
        LogisticRegressionProblem {
            x,
            target,
            alpha: self.alpha,
        }
    }

    /// Create the LBFGS solver using MoreThuenteLineSearch and set gradient
    /// tolerance.
    fn setup_solver(&self) -> LBFGSType<F, D> {
        let linesearch = MoreThuenteLineSearch::new();
        LBFGS::new(linesearch, 10).with_tol_grad(self.gradient_tolerance)
    }

    /// Run the LBFGS solver until it converges or runs out of iterations.
    fn run_solver<P: SolvableProblem>(
        &self,
        problem: P,
        solver: P::Solver,
        init_params: P::Param,
    ) -> Result<ArgminResult<P>> {
        Executor::new(problem, solver, init_params)
            .max_iters(self.max_iterations)
            .run()
            .map_err(|err| err.into())
    }
}

impl<'a, C: 'a + Ord + Clone, F: Float, D: Data<Elem = F>, T: AsTargets<Elem = C>>
    Fit<ArrayBase<D, Ix2>, T, Error> for LogisticRegression<F>
{
    type Object = FittedLogisticRegression<F, C>;

    /// Given a 2-dimensional feature matrix array `x` with shape
    /// (n_samples, n_features) and an array of target classes to predict,
    /// create a `FittedLinearRegression` object which allows making
    /// predictions.
    ///
    /// The array of target classes `y` must have exactly two distinct
    /// values, (e.g. 0.0 and 1.0, 0 and 1, "cat" and "dog", ...), which
    /// represent the two different classes the model is supposed to predict.
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
        let problem = self.setup_problem(x, target);
        let solver = self.setup_solver();
        let init_params = self.setup_init_params(x.ncols());
        let result = self.run_solver(problem, solver, ArgminParam(init_params))?;

        let params = result.state().best_param.as_array();
        let (w, intercept) = convert_params(x.nrows(), params);
        Ok(FittedLogisticRegression::new(
            intercept.into_scalar(),
            w,
            labels,
        ))
    }
}

impl<'a, C: 'a + Ord + Clone, F: Float, D: Data<Elem = F>, T: AsTargets<Elem = C>>
    Fit<ArrayBase<D, Ix2>, T, Error> for MultiLogisticRegression<F>
{
    type Object = MultiFittedLogisticRegression<F, C>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let (x, y) = (dataset.records(), dataset.targets());
        let (classes, target) = label_classes_multi(y)?;
        self.validate_data(x, &target)?;
        let problem = self.setup_problem(x, target);
        let solver = self.setup_solver();
        let init_params = self.setup_init_params((x.ncols(), classes.len()));
        let result = self.run_solver(problem, solver, ArgminParam(init_params))?;

        let params = result.state().best_param.as_array();
        let (w, intercept) = convert_params(x.nrows(), params);
        Ok(MultiFittedLogisticRegression::new(intercept, w, classes))
    }
}

/// Identify the distinct values of the classes  `y` and associate
/// the target labels `-1.0` and `1.0` to it. -1.0 always labels the
/// smaller class (by PartialOrd) and 1.0 always labels the larger
/// class.
///
/// It is an error to have more than two classes.
fn label_classes<F, T, C>(y: T) -> Result<(ClassLabels<F, C>, Array1<F>)>
where
    F: Float,
    T: AsTargets<Elem = C>,
    C: Ord + Clone,
{
    let y_single_target = y.try_single_target()?;
    let mut classes: Vec<&C> = vec![];
    let mut target_vec = vec![];
    let mut use_negative_label: bool = true;
    for item in y_single_target {
        if let Some(last_item) = classes.last() {
            if *last_item != item {
                use_negative_label = !use_negative_label;
            }
        }
        if !classes.contains(&item) {
            classes.push(item);
        }
        target_vec.push(if use_negative_label {
            F::NEGATIVE_LABEL
        } else {
            F::POSITIVE_LABEL
        });
    }
    if classes.len() != 2 {
        return Err(Error::WrongNumberOfClasses);
    }
    let mut target_array = Array1::from(target_vec);
    let labels = if classes[0] < classes[1] {
        (F::NEGATIVE_LABEL, F::POSITIVE_LABEL)
    } else {
        // If we found the larger class first, flip the sign in the target
        // vector, so that -1.0 is always the label for the smaller class
        // and 1.0 the label for the larger class
        target_array *= -F::one();
        (F::POSITIVE_LABEL, F::NEGATIVE_LABEL)
    };
    Ok((
        vec![
            ClassLabel {
                class: classes[0].clone(),
                label: labels.0,
            },
            ClassLabel {
                class: classes[1].clone(),
                label: labels.1,
            },
        ],
        target_array,
    ))
}

fn label_classes_multi<F, T, C>(y: T) -> Result<(Vec<C>, Array2<F>)>
where
    F: Float,
    T: AsTargets<Elem = C>,
    C: Ord + Clone,
{
    let y_single_target = y.try_single_target()?;
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
) -> (Array<F, D>, Array<F, D::Smaller>) {
    let nrows = w.shape()[0];
    if n_features == nrows {
        (w.to_owned(), Array::zeros(w.raw_dim().remove_axis(Axis(0))))
    } else if n_features + 1 == nrows {
        (
            w.slice_axis(Axis(0), Slice::from(..n_features)).to_owned(),
            w.index_axis(Axis(0), n_features).to_owned(),
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

/// The logistic function
fn logistic<F: linfa::Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

/// A numerically stable version of the log of the logistic function.
///
/// Taken from scikit-learn
/// https://github.com/scikit-learn/scikit-learn/blob/0.23.1/sklearn/utils/_logistic_sigmoid.pyx
///
/// See the blog post describing this implementation:
/// http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
fn log_logistic<F: linfa::Float>(x: F) -> F {
    if x > F::zero() {
        -(F::one() + (-x).exp()).ln()
    } else {
        x - (F::one() + x.exp()).ln()
    }
}

/// Finds the log of the sum of exponents across a specific axis in a numerically stable way. More
/// specifically, computes `ln(exp(x1) + exp(x2) + exp(e3) + ...)` across an axis.
///
/// Based off this implementation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
fn log_sum_exp<F: linfa::Float, A: Data<Elem = F>>(
    m: &ArrayBase<A, Ix2>,
    axis: Axis,
) -> Array<F, Ix1> {
    // Find max value of the array
    let max = m.iter().copied().reduce(F::max).unwrap();
    // Computes `max + ln(exp(x1-max) + exp(x2-max) + exp(x3-max) + ...)`, which is equal to the
    // log_sum_exp formula
    let reduced = m.fold_axis(axis, F::zero(), |acc, elem| *acc + (*elem - max).exp());
    reduced.mapv_into(|e| e.ln() + max)
}

/// Computes `exp(n - max) / sum(exp(n- max))`, which is a numerically stable version of softmax
fn softmax_inplace<F: linfa::Float, A: DataMut<Elem = F>>(v: &mut ArrayBase<A, Ix1>) {
    let max = v.iter().copied().reduce(F::max).unwrap();
    v.mapv_inplace(|n| (n - max).exp());
    let sum = v.sum();
    v.mapv_inplace(|n| n / sum);
}

/// Computes the logistic loss assuming the training labels $y \in {-1, 1}$
///
/// Because the logistic function fullfills $\sigma(-z) = 1 - \sigma(z)$
/// we can write $P(y=1|z) = \sigma(z) = \sigma(yz)$ and
/// $P(y=-1|z) = 1 - P(y=1|z) = 1 - \sigma(z) = \sigma(-z) = \sigma(yz)$, so
/// $P(y|z) = \sigma(yz)$ for both $y=1$ and $y=-1$.
///
/// Thus, the log loss can be written as
/// $$-\sum_{i=1}^{N} \log(\sigma(y_i z_i)) + \frac{\alpha}{2}\text{params}^T\text{params}$$
fn logistic_loss<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array1<F>,
    alpha: F,
    w: &Array1<F>,
) -> F {
    let n_features = x.shape()[1];
    let (params, intercept) = convert_params(n_features, w);
    let mut yz = (x.dot(&params) + intercept) * y;
    yz.mapv_inplace(log_logistic);
    -yz.sum() + F::cast(0.5) * alpha * params.dot(&params)
}

/// Compute the log of probabilities, which is `log(softmax(H))`, along with `W`.
/// `Y` is the output (n_samples * n_classes), `X` is the input (n_samples * n_features), `W` is the
/// params (n_features * n_classes), `b` is the intercept vector (n_classes).
fn multi_logistic_prob_params<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    w: &Array2<F>, // This parameter includes `W` and `b`
) -> (Array2<F>, Array2<F>) {
    let n_features = x.shape()[1];
    let (params, intercept) = convert_params(n_features, w);
    // Compute H
    let h = x.dot(&params) + intercept;
    // This computes `H - log(sum(exp(H)))`, which is equal to
    // `log(softmax(H)) = log(exp(H) / sum(exp(H)))`
    let log_prob = &h - log_sum_exp(&h, Axis(1));
    (log_prob, params)
}

/// Computes loss function of `-sum(Y * log(softmax(H))) + alpha/2 * norm(W)`, where H is `X . W + b`.
fn multi_logistic_loss<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array2<F>,
    alpha: F,
    w: &Array2<F>,
) -> F {
    let (log_prob, params) = multi_logistic_prob_params(x, w);
    // Calculate loss
    // XXX Should we divide cost by n_samples???
    -elem_dot(&log_prob, y) + F::cast(0.5) * alpha * elem_dot(&params, &params)
}

/// Computes the gradient of the logistic loss function
fn logistic_grad<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array1<F>,
    alpha: F,
    w: &Array1<F>,
) -> Array1<F> {
    let n_features = x.shape()[1];
    let (params, intercept) = convert_params(n_features, w);
    let mut yz = (x.dot(&params) + intercept) * y;
    yz.mapv_inplace(logistic);
    yz -= F::one();
    yz *= y;
    if w.len() == n_features + 1 {
        let mut grad = Array::zeros(w.len());
        grad.slice_mut(s![..n_features])
            .assign(&(x.t().dot(&yz) + &(params * alpha)));
        grad[n_features] = yz.sum();
        grad
    } else {
        x.t().dot(&yz) + &(params * alpha)
    }
}

/// Computes multinomial gradients for `W` and `b` and combine them.
/// Gradient for `W` is `Xt . (softmax(H) - Y) + alpha * W`.
/// Gradient for `b` is `sum(softmax(H) - Y)`.
fn multi_logistic_grad<F: Float, A: Data<Elem = F>>(
    x: &ArrayBase<A, Ix2>,
    y: &Array2<F>,
    alpha: F,
    w: &Array2<F>,
) -> Array2<F> {
    let (log_prob, mut params) = multi_logistic_prob_params(x, w);
    let (n_features, n_classes) = params.dim();
    let intercept = x.ncols() > n_features;
    let mut grad = Array::zeros((n_features + intercept as usize, n_classes));

    // This value is `softmax(H)`
    let prob = log_prob.mapv_into(num_traits::Float::exp);
    let diff = prob - y;
    // Compute gradient for `W` and place it at start of the grad matrix
    let mut dw = x.t().dot(&diff);
    params *= alpha;
    dw += &params;
    grad.slice_mut(s![..n_features, ..]).assign(&dw);
    // Compute gradient for `b` and place it at end of grad matrix
    if intercept {
        grad.row_mut(n_features).assign(&diff.sum_axis(Axis(0)));
    }
    // XXX should we divide by n_samples??
    grad
}

/// A fitted logistic regression which can make predictions
#[derive(PartialEq, Debug)]
pub struct FittedLogisticRegression<F: Float, C: PartialOrd + Clone> {
    threshold: F,
    intercept: F,
    params: Array1<F>,
    labels: ClassLabels<F, C>,
}

impl<F: Float, C: PartialOrd + Clone> FittedLogisticRegression<F, C> {
    fn new(
        intercept: F,
        params: Array1<F>,
        labels: ClassLabels<F, C>,
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

    /// Given a feature matrix, predict the probabilities that a sample
    /// should be classified as the larger of the two classes learned when the
    /// model was fitted.
    pub fn predict_probabilities<A: Data<Elem = F>>(&self, x: &ArrayBase<A, Ix2>) -> Array1<F> {
        let mut probs = x.dot(&self.params) + self.intercept;
        probs.mapv_inplace(logistic);
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

        let pos_class = class_from_label(&self.labels, F::POSITIVE_LABEL);
        let neg_class = class_from_label(&self.labels, F::NEGATIVE_LABEL);
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

/// A fitted logistic regression which can make predictions
#[derive(PartialEq, Debug)]
pub struct MultiFittedLogisticRegression<F: Float, C: PartialOrd + Clone> {
    threshold: F,
    intercept: Array1<F>,
    params: Array2<F>,
    classes: Vec<C>,
}

impl<F: Float, C: PartialOrd + Clone> MultiFittedLogisticRegression<F, C> {
    fn new(intercept: Array1<F>, params: Array2<F>, classes: Vec<C>) -> Self {
        Self {
            threshold: F::cast(0.5),
            intercept,
            params,
            classes,
        }
    }

    /// Set the probability threshold for which the 'positive' class will be
    /// predicted. Defaults to 0.5.
    pub fn set_threshold(mut self, threshold: F) -> Self {
        if threshold < F::zero() || threshold > F::one() {
            panic!("FittedLogisticRegression::set_threshold: threshold needs to be between 0.0 and 1.0");
        }
        self.threshold = threshold;
        self
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
        probs
            .rows_mut()
            .into_iter()
            .for_each(|mut row| softmax_inplace(&mut row));
        probs
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

#[derive(PartialEq, Debug, Clone)]
struct ClassLabel<F: Float, C: PartialOrd> {
    class: C,
    label: F,
}

type ClassLabels<F, C> = Vec<ClassLabel<F, C>>;

fn class_from_label<F: Float, C: PartialOrd + Clone>(labels: &[ClassLabel<F, C>], label: F) -> C {
    labels
        .iter()
        .find(|cl| cl.label == label)
        .unwrap()
        .class
        .clone()
}

/// Internal representation of a logistic regression problem.
/// This data structure exists to be handed to Argmin.
struct LogisticRegressionProblem<'a, F: Float, A: Data<Elem = F>, D: Dimension> {
    x: &'a ArrayBase<A, Ix2>,
    target: Array<F, D>,
    alpha: F,
}

type LogisticRegressionProblem1<'a, F, A> = LogisticRegressionProblem<'a, F, A, Ix1>;
type LogisticRegressionProblem2<'a, F, A> = LogisticRegressionProblem<'a, F, A, Ix2>;

impl<'a, F: Float, A: Data<Elem = F>> ArgminOp for LogisticRegressionProblem1<'a, F, A> {
    type Param = ArgminParam<F, Ix1>;
    type Output = F;
    type Hessian = ();
    type Jacobian = Array1<F>;
    type Float = F;

    /// Apply the cost function to a parameter `p`
    fn apply(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let w = p.as_array();
        Ok(logistic_loss(self.x, &self.target, self.alpha, w))
    }

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> std::result::Result<Self::Param, argmin::core::Error> {
        let w = p.as_array();
        Ok(ArgminParam(logistic_grad(
            self.x,
            &self.target,
            self.alpha,
            w,
        )))
    }
}

impl<'a, F: Float, A: Data<Elem = F>> ArgminOp for LogisticRegressionProblem2<'a, F, A> {
    type Param = ArgminParam<F, Ix2>;
    type Output = F;
    type Hessian = ();
    type Jacobian = Array1<F>;
    type Float = F;

    /// Apply the cost function to a parameter `p`
    fn apply(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let w = p.as_array();
        Ok(multi_logistic_loss(self.x, &self.target, self.alpha, w))
    }

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> std::result::Result<Self::Param, argmin::core::Error> {
        let w = p.as_array();
        Ok(ArgminParam(multi_logistic_grad(
            self.x,
            &self.target,
            self.alpha,
            w,
        )))
    }
}

trait SolvableProblem: ArgminOp + Sized {
    type Solver: Solver<Self>;
}

impl<'a, F: Float, A: Data<Elem = F>> SolvableProblem for LogisticRegressionProblem1<'a, F, A> {
    type Solver = LBFGSType1<F>;
}

impl<'a, F: Float, A: Data<Elem = F>> SolvableProblem for LogisticRegressionProblem2<'a, F, A> {
    type Solver = LBFGSType2<F>;
}

#[cfg(test)]
mod test {
    extern crate linfa;

    use super::*;
    use approx::{assert_abs_diff_eq, AbsDiffEq};
    use linfa::prelude::*;
    use ndarray::{array, Array2};

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
        let ws = vec![
            array![0.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 0.0],
            array![1.0, 1.0],
            array![0.0, -1.0],
            array![-1.0, 0.0],
            array![-1.0, -1.0],
        ];
        let alphas = vec![0.0, 1.0, 10.0];
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
            assert_abs_diff_eq!(logistic_loss(&x, &y, alpha, w), *exp);
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
        let ws = vec![
            array![0.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 0.0],
            array![1.0, 1.0],
            array![0.0, -1.0],
            array![-1.0, 0.0],
            array![-1.0, -1.0],
        ];
        let alphas = vec![0.0, 1.0, 10.0];
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
            let actual = logistic_grad(&x, &y, alpha, w);
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
            dataset.targets().try_single_target().unwrap()
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
            dataset.targets().try_single_target().unwrap()
        );
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
        assert!(res.intercept().abs_diff_eq(&-4.124, 1e-3));
        assert!(res.params().abs_diff_eq(&array![1.181], 1e-3));
        assert_eq!(
            &res.predict(dataset.records()),
            dataset.targets().try_single_target().unwrap()
        );
    }

    #[test]
    fn rejects_multi_target() {
        let log_reg = LogisticRegression::default();
        let x = array![[0.01], [1.0], [-1.0], [-0.01]];
        let y = array![[0, 0], [0, 0], [0, 0], [0, 0]];
        let res = log_reg.fit(&Dataset::new(x, y));
        assert_eq!(
            res.unwrap_err().to_string(),
            "multiple targets not supported".to_string()
        );
    }

    #[test]
    fn rejects_mismatching_x_y() {
        let log_reg = LogisticRegression::default();
        let x = array![[-1.0], [-0.01], [0.01]];
        let y = array![0, 0, 1, 1];
        let res = log_reg.fit(&Dataset::new(x, y));
        assert_eq!(
            res.unwrap_err().to_string(),
            "Expected `x` and `y` to have same number of rows, got 3 != 4".to_string()
        );
    }

    #[test]
    fn rejects_inf_values() {
        let infs = vec![std::f64::INFINITY, std::f64::NEG_INFINITY, std::f64::NAN];
        let inf_xs: Vec<_> = infs.iter().map(|&inf| array![[1.0], [inf]]).collect();
        let log_reg = LogisticRegression::default();
        let normal_x = array![[-1.0], [1.0]];
        let y = array![0, 1];
        let expected = "Values must be finite and not `Inf`, `-Inf` or `NaN`".to_string();
        for inf_x in &inf_xs {
            let res = log_reg.fit(&Dataset::new(inf_x.to_owned(), y.to_owned()));
            assert_eq!(res.unwrap_err().to_string(), expected);
        }
        for inf in &infs {
            let log_reg = LogisticRegression::default().alpha(*inf);
            let res = log_reg.fit(&Dataset::new(normal_x.to_owned(), y.to_owned()));
            assert_eq!(res.unwrap_err().to_string(), expected);
        }
        let mut non_positives = infs;
        non_positives.push(-1.0);
        non_positives.push(0.0);
        for inf in &non_positives {
            let log_reg = LogisticRegression::default().gradient_tolerance(*inf);
            let res = log_reg.fit(&Dataset::new(normal_x.to_owned(), y.to_owned()));
            assert_eq!(
                res.unwrap_err().to_string(),
                "gradient_tolerance must be a positive, finite number"
            );
        }
    }

    #[test]
    fn validates_initial_params() {
        let infs = vec![std::f64::INFINITY, std::f64::NEG_INFINITY, std::f64::NAN];
        let normal_x = array![[-1.0], [1.0]];
        let normal_y = array![0, 1];
        let dataset = Dataset::new(normal_x, normal_y);
        let expected = "Initial parameter guess must be finite".to_string();
        for inf in &infs {
            let log_reg = LogisticRegression::default().initial_params(array![*inf]);
            let res = log_reg.fit(&dataset);
            assert_eq!(res.unwrap_err().to_string(), expected);
        }
        {
            let log_reg = LogisticRegression::default().initial_params(array![0.0, 0.0]);
            let res = log_reg.fit(&dataset);
            assert_eq!(res.unwrap_err().to_string(), "Size of initial parameter guess must be the same as the number of columns in the feature matrix `x`".to_string());
        }
        {
            let log_reg = LogisticRegression::default()
                .with_intercept(false)
                .initial_params(array![0.0, 0.0]);
            let res = log_reg.fit(&dataset);
            assert_eq!(
                res.unwrap_err().to_string(),
                "Initial parameter should be smaller when there's no intercept".to_string()
            );
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
            dataset.targets().try_single_target().unwrap()
        );
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
            dataset.targets().try_single_target().unwrap()
        );
    }
}
