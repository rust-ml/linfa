use linfa::{
    error::{Error, Result},
    ParamGuard,
};
use rand::rngs::ThreadRng;
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnsembleLearnerValidParams<P, R> {
    /// The number of models in the ensemble
    pub ensemble_size: usize,
    /// The proportion of the total number of training samples that should be given to each model for training
    pub bootstrap_proportion: f64,
    /// The model parameters for the base model
    pub model_params: P,
    pub rng: R,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnsembleLearnerParams<P, R>(EnsembleLearnerValidParams<P, R>);

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
            model_params,
            rng,
        })
    }

    pub fn ensemble_size(mut self, size: usize) -> Self {
        self.0.ensemble_size = size;
        self
    }

    pub fn bootstrap_proportion(mut self, proportion: f64) -> Self {
        self.0.bootstrap_proportion = proportion;
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
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
