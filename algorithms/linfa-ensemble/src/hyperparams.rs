use linfa::{
    error::{Error, Result},
    ParamGuard,
};
use linfa_trees::DecisionTreeParams;
use rand::rngs::ThreadRng;
use rand::Rng;

/// The set of valid hyper-parameters that can be specified for the fitting procedure of the
/// [Ensemble Learner](crate::EnsembleLearner).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnsembleLearnerValidParams<P, R> {
    /// The number of models in the ensemble
    pub ensemble_size: usize,
    /// The proportion of the total number of training samples that should be given to each model for training
    pub bootstrap_proportion: f64,
    /// The proportion of the total number of training features that should be given to each model for training
    pub feature_proportion: f64,
    /// The model parameters for the base model
    pub model_params: P,
    pub rng: R,
}

/// A helper struct for building a set of [Ensemble Learner](crate::EnsembleLearner) hyper-parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnsembleLearnerParams<P, R>(EnsembleLearnerValidParams<P, R>);

/// A helper struct for building a set of [Random Forest](crate::RandomForest) hyper-parameters.
pub type RandomForestParams<F, L, R> = EnsembleLearnerParams<DecisionTreeParams<F, L>, R>;

impl<P> EnsembleLearnerParams<P, ThreadRng> {
    pub fn new(model_params: P) -> EnsembleLearnerParams<P, ThreadRng> {
        Self::new_fixed_rng(model_params, rand::thread_rng())
    }
}

impl<P, R: Rng + Clone> EnsembleLearnerParams<P, R> {
    pub fn new_fixed_rng(model_params: P, rng: R) -> EnsembleLearnerParams<P, R> {
        Self(EnsembleLearnerValidParams {
            ensemble_size: 1,
            bootstrap_proportion: 1.0,
            feature_proportion: 1.0,
            model_params,
            rng,
        })
    }

    /// Specifies the number of models to fit in the ensemble.
    pub fn ensemble_size(mut self, size: usize) -> Self {
        self.0.ensemble_size = size;
        self
    }

    /// Sets the proportion of the total number of training samples that should be given to each model for training
    ///
    /// Note that the `proportion` should be in the interval (0, 1] in order to pass the  
    /// parameter validation check.
    pub fn bootstrap_proportion(mut self, proportion: f64) -> Self {
        self.0.bootstrap_proportion = proportion;
        self
    }

    /// Sets the proportion of the total number of training features that should be given to each model for training
    ///
    /// Note that the `proportion` should be in the interval (0, 1] in order to pass the
    /// parameter validation check.
    pub fn feature_proportion(mut self, proportion: f64) -> Self {
        self.0.feature_proportion = proportion;
        self
    }
}

impl<P, R> ParamGuard for EnsembleLearnerParams<P, R> {
    type Checked = EnsembleLearnerValidParams<P, R>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.bootstrap_proportion > 1.0 || self.0.bootstrap_proportion <= 0.0 {
            Err(Error::Parameters(format!(
                "Bootstrap proportion should be greater than zero and less than or equal to one, but was {}",
                self.0.bootstrap_proportion
            )))
        } else if self.0.ensemble_size < 1 {
            Err(Error::Parameters(format!(
                "Ensemble size should be less than one, but was {}",
                self.0.ensemble_size
            )))
        } else if self.0.feature_proportion > 1.0 || self.0.feature_proportion <= 0.0 {
            Err(Error::Parameters(format!(
                "Feature proportion should be greater than zero and less than or equal to one, but was {}",
                self.0.feature_proportion
            )))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
