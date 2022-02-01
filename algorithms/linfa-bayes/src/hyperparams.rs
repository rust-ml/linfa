use crate::NaiveBayesError;
use linfa::{Float, ParamGuard};
use std::marker::PhantomData;

/// A verified hyper-parameter set ready for the estimation of a [Gaussian Naive Bayes model](crate::gaussian_nb::GaussianNb).
///
/// See [`GaussianNb`](crate::gaussian_nb::GaussianNb) for information on the model and [`GaussianNbParams`](crate::hyperparams::GaussianNbParams) for information on hyperparameters.
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

/// A hyper-parameter set during construction for a [Gaussian Naive Bayes model](crate::gaussian_nb::GaussianNb).
///
/// The parameter set can be verified into a
/// [`GaussianNbValidParams`](crate::hyperparams::GaussianNbValidParams) by calling
/// [ParamGuard::check](Self::check). It is also possible to directly fit a model with
/// [Fit::fit](linfa::traits::Fit::fit) or
/// [FitWith::fit_with](linfa::traits::FitWith::fit_with) which implicitely verifies the parameter set
/// prior to the model estimation and forwards any error.
/// 
/// See [`GaussianNb`](crate::gaussian_nb::GaussianNb) for information on the model.
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

/// A verified hyper-parameter set ready for the estimation of a [Multinomial Naive Bayes model](crate::multinomial_nb::MultinomialNb).
///
/// See [`MultinomialNb`](crate::multinomial_nb::MultinomialNb) for information on the model and [`MultinomialNbParams`](crate::hyperparams::MultinomialNbParams) for information on hyperparameters.
#[derive(Debug)]
pub struct MultinomialNbValidParams<F, L> {
    // Required for calculation stability
    alpha: F,
    // Phantom data for label type
    label: PhantomData<L>,
}

impl<F: Float, L> MultinomialNbValidParams<F, L> {
    /// Get the variance smoothing
    pub fn alpha(&self) -> F {
        self.alpha
    }
}

/// A hyper-parameter set during construction for a [Multinomial Naive Bayes model](crate::multinomial_nb::MultinomialNb). 
///
/// The parameter set can be verified into a
/// [`MultinomialNbValidParams`](crate::hyperparams::MultinomialNbValidParams) by calling
/// [ParamGuard::check](Self::check). It is also possible to directly fit a model with
/// [Fit::fit](linfa::traits::Fit::fit) or
/// [FitWith::fit_with](linfa::traits::FitWith::fit_with) which implicitely verifies the parameter set
/// prior to the model estimation and forwards any error.
///
/// See [`MultinomialNb`](crate::multinomial_nb::MultinomialNb) for information on the model.
/// 
/// # Parameters
/// | Name | Default | Purpose | Range |
/// | :--- | :--- | :---| :--- |
/// | [alpha](Self::alpha) | `1` | Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing) | `[0, inf)` |
///
/// # Errors
///
/// The following errors can come from invalid hyper-parameters:
///
/// Returns [`InvalidSmoothing`](NaiveBayesError::InvalidSmoothing) if the smoothing
/// parameter is negative.
///
pub struct MultinomialNbParams<F, L>(MultinomialNbValidParams<F, L>);

impl<F: Float, L> Default for MultinomialNbParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, L> MultinomialNbParams<F, L> {
    /// Create new [MultinomialNbParams] set with default values for its parameters
    pub fn new() -> Self {
        Self(MultinomialNbValidParams {
            alpha: F::cast(1),
            label: PhantomData,
        })
    }

    /// Specifies the portion of the largest variance of all the features that
    /// is added to the variance for calculation stability
    pub fn alpha(mut self, alpha: F) -> Self {
        self.0.alpha = alpha;
        self
    }
}

impl<F: Float, L> ParamGuard for MultinomialNbParams<F, L> {
    type Checked = MultinomialNbValidParams<F, L>;
    type Error = NaiveBayesError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.alpha.is_negative() {
            Err(NaiveBayesError::InvalidSmoothing(
                self.0.alpha.to_f64().unwrap(),
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
