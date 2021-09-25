use crate::NaiveBayesError;
use linfa::{Float, ParamGuard};
use std::marker::PhantomData;

/// A verified hyper-parameter set ready for the estimation of a Gaussian Naive Bayes model
///
/// See [`GaussianNbParams`](crate::hyperparams::GaussianNbParams) for more informations.
#[derive(Debug)]
pub struct GaussianNbValidParams<F, L> {
    // Required for calculation stability
    var_smoothing: F,
    // Phantom data for label type
    label: PhantomData<L>,
}

impl<F: Float, L> GaussianNbValidParams<F, L> {
    /// Get the variance smoothing
    pub fn var_smoothing(&self) -> F {
        self.var_smoothing
    }
}

/// A hyper-parameter set during construction
///
/// The parameter set can be verified into a
/// [`GaussianNbValidParams`](crate::hyperparams::GaussianNbValidParams) by calling
/// [ParamGuard::check](Self::check). It is also possible to directly fit a model with
/// [Fit::fit](linfa::traits::Fit::fit) or
/// [FitWith::fit_with](linfa::traits::FitWith::fit_with) which implicitely verifies the parameter set
/// prior to the model estimation and forwards any error.
///
/// # Parameters
/// | Name | Default | Purpose | Range |
/// | :--- | :--- | :---| :--- |
/// | [var_smoothing](Self::var_smoothing) | `1e-9` | Stabilize variance calculation if ratios are small in update step | `[0, inf)` |
///
/// # Errors
///
/// The following errors can come from invalid hyper-parameters:
///
/// Returns [`InvalidSmoothing`](NaiveBayesError::InvalidSmoothing) if the smoothing
/// parameter is negative.
///
/// # Example
///
/// ```rust
/// use linfa_bayes::{GaussianNbParams, GaussianNbValidParams, Result};
/// use linfa::prelude::*;
/// use ndarray::array;
///
/// let x = array![
///     [-2., -1.],
///     [-1., -1.],
///     [-1., -2.],
///     [1., 1.],
///     [1., 2.],
///     [2., 1.]
/// ];
/// let y = array![1, 1, 1, 2, 2, 2];
/// let ds = DatasetView::new(x.view(), y.view());
///
/// // create a new parameter set with variance smoothing equals `1e-5`
/// let unchecked_params = GaussianNbParams::new()
///     .var_smoothing(1e-5);
///
/// // fit model with unchecked parameter set
/// let model = unchecked_params.fit(&ds)?;
///
/// // transform into a verified parameter set
/// let checked_params = unchecked_params.check()?;
///
/// // update model with the verified parameters, this only returns
/// // errors originating from the fitting process
/// let model = checked_params.fit_with(Some(model), &ds)?;
/// # Result::Ok(())
/// ```
pub struct GaussianNbParams<F, L>(GaussianNbValidParams<F, L>);

impl<F: Float, L> Default for GaussianNbParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, L> GaussianNbParams<F, L> {
    /// Create new [GaussianNbParams] set with default values for its parameters
    pub fn new() -> Self {
        Self(GaussianNbValidParams {
            var_smoothing: F::cast(1e-9),
            label: PhantomData,
        })
    }

    /// Specifies the portion of the largest variance of all the features that
    /// is added to the variance for calculation stability
    pub fn var_smoothing(mut self, var_smoothing: F) -> Self {
        self.0.var_smoothing = var_smoothing;
        self
    }
}

impl<F: Float, L> ParamGuard for GaussianNbParams<F, L> {
    type Checked = GaussianNbValidParams<F, L>;
    type Error = NaiveBayesError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.var_smoothing.is_negative() {
            Err(NaiveBayesError::InvalidSmoothing(
                self.0.var_smoothing.to_f64().unwrap(),
            ))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
