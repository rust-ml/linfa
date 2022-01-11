use crate::NaiveBayesError;
use linfa::{Float, ParamGuard};
use std::marker::PhantomData;

// Hyperparameters of Multinomial Naive Bayes
#[derive(Debug, Clone)]
pub struct MultinomialNbParams<F>{pub alpha: F}
// Hyperparameters of Gaussian Naive Bayes
#[derive(Debug, Clone)]
pub struct GaussianNbParams<F>{pub var_smoothing: F}


/// Container for hyperparameters of any Naive Bayes model
#[derive(Debug, Clone)]
pub enum NbModel<F> {
    /// Multinomial Naive Bayes
    MultinomialNb(MultinomialNbParams<F>),
    /// Gaussian Naive Bayes
    GaussianNb(GaussianNbParams<F>),
}

/// A verified hyper-parameter set ready for the estimation of a Naive Bayes model
/// 
/// The specific set of hyper-parameters is stored in the [`NbValidParams::model`](NbValidParams::model) 
/// field and depends on the type of Naive Bayes used. 
/// See [`NbParams`](crate::hyperparams::NbParams) for more informations.
#[derive(Debug)]
pub struct NbValidParams<F, L> {
    /// Hyperparameters of a Naive Bayes model
    pub model: NbModel<F>,
    /// Phantom data for label type
    label: PhantomData<L>,
}


/// A hyper-parameter set during construction
///
/// The parameter set can be verified into a
/// [`NbValidParams`](crate::hyperparams::NbValidParams) by calling
/// [ParamGuard::check](Self::check). It is also possible to directly fit a model with
/// [Fit::fit](linfa::traits::Fit::fit) or
/// [FitWith::fit_with](linfa::traits::FitWith::fit_with) which implicitely verifies
/// the parameter set prior to the model estimation and forwards any error.
/// 
/// # Parameters
/// 
/// [`NbParams`](NbParams) can contain different hyper-parameter sets for different
/// kinds of Naive Bayes, depending on the chosen variant of enum 
/// [`NbValidParams::model`](crate::hyperparams::NbValidParams::model).
/// 
/// ## Parameters for Gaussian Naive Bayes
/// | Name | Default | Purpose | Range |
/// | :--- | :--- | :---| :--- |
/// | [var_smoothing](Self::var_smoothing) | `1e-9` | Stabilize variance calculation if ratios are small in update step | `[0, inf)` |
/// ## Parameters for Multinomial Naive Bayes
/// | Name | Default | Purpose | Range |
/// | :--- | :--- | :---| :--- |
/// | [alpha](Self::alpha) | `1` | Additive (Laplace/Lidstone) smoothing parameter. Regularizes log probabilities if certain feature/class combinations don't occur | `[0, inf)` |
/// 
/// # Errors
///
/// The following errors can come from invalid hyper-parameters:
///
/// Returns [`InvalidSmoothing`](NaiveBayesError::InvalidSmoothing) if the smoothing
/// parameter is negative.
/// 
/// # Examples
/// ## Example of Gaussian Naive Bayes
///
/// ```rust
/// use linfa_bayes::{NbParams, NbValidParams, Result};
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
/// // create a new parameter set with variance smoothing equal `1e-5`
/// let unchecked_params = NbParams::new().gaussian()
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
/// 
/// ## Example of Multinomial Naive Bayes 
///
/// 
/// ```rust
///
/// 
/// 
/// use linfa_bayes::{NbParams, NbValidParams, Result};
/// use linfa::prelude::*;
/// use ndarray::array;
///
/// let x = array![
///     [1., 0.],
///     [2., 0.],
///     [3., 0.],
///     [0., 1.],
///     [0., 2.],
///     [0., 3.]
/// ];
/// let y = array![1, 1, 1, 2, 2, 2];
/// let ds = DatasetView::new(x.view(), y.view());
///
/// // create a new parameter set with smoothing parameter alpha equal `1.5`
/// let unchecked_params = NbParams::new().multinomial()
///     .alpha(1.5);
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
#[derive(Debug)]
pub struct NbParams<F, L>(NbValidParams<F, L>);

impl<F: Float> Default for NbModel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> NbModel<F> {
    pub fn new() -> Self {
        NbModel::<F>::MultinomialNb(MultinomialNbParams {alpha : F::one()})
    }
}

impl<F: Float, L> NbParams<F, L> {
    /// Create a new [NbParams] set for default Naive Bayes model with default parameter values
    pub fn new() -> Self {
        Self(NbValidParams::<F, L> {
            model: NbModel::default(),
            label: PhantomData,
        })
    }
    
    /// Create Gaussian Naive Bayes model
    pub fn gaussian(mut self) -> Self {
        self.0.model = NbModel::GaussianNb(GaussianNbParams {var_smoothing : F::cast(1e-9)});
        self
    }
    
    /// Create Multinomial Naive Bayes model
    pub fn multinomial(mut self) -> Self {
        self.0.model = NbModel::MultinomialNb(MultinomialNbParams {alpha : F::cast(1)});
        self
    }

    /// Set value of smoothing parameter for Gaussian Naive Bayes model
    pub fn var_smoothing(self, var_smoothing: F) -> Self {
        match self.0.model {
            NbModel::GaussianNb(mut nb_model) => {nb_model.var_smoothing = var_smoothing;
                Self(NbValidParams{model: NbModel::GaussianNb(nb_model), label: PhantomData})},
            NbModel::MultinomialNb(_) => panic!("Trying to set smoothing paramater on wrong model type - expected Gaussian Naive Bayes")
        }       
            }
    
    /// Set value of alpha parameter for Multinomial Naive Bayes model
    pub fn alpha(self, alpha: F) -> Self {
        match self.0.model {
            NbModel::MultinomialNb(mut nb_model) => {nb_model.alpha = alpha;
                Self(NbValidParams{model: NbModel::MultinomialNb(nb_model), label: PhantomData})},
            NbModel::GaussianNb(_) => panic!("Trying to set paramater alpha on wrong model type - expected Multinomial Naive Bayes")
        }       
            }


}

impl<F: Float, L> Default for NbParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, L> ParamGuard for NbParams<F, L>{
    type Checked = NbValidParams<F, L>;
    type Error = NaiveBayesError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
            match &self.0.model {
                NbModel::MultinomialNb(nb_model) => nb_model.check_ref()?,
                NbModel::GaussianNb(nb_model) => nb_model.check_ref()?
            };
            Ok(&self.0)
        }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
