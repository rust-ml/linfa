use crate::gaussian_mixture::errors::{GmmError, Result};
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

use linfa::{Float, ParamGuard};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug, PartialEq)]
/// A specifier for the type of the relation between components' covariances.
pub enum GmmCovarType {
    /// each component has its own general covariance matrix
    Full,
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
/// A specifier for the method used for the initialization of the fitting algorithm of GMM
pub enum GmmInitMethod {
    /// GMM fitting algorithm is initalized with the esult of the [KMeans](struct.KMeans.html) clustering.
    KMeans,
    /// GMM fitting algorithm is initialized randomly.
    Random,
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [GMM algorithm](struct.GaussianMixtureModel.html).
pub struct GmmValidParams<F: Float, R: Rng> {
    n_clusters: usize,
    covar_type: GmmCovarType,
    tolerance: F,
    reg_covar: F,
    n_runs: u64,
    max_n_iter: u64,
    init_method: GmmInitMethod,
    rng: R,
}

impl<F: Float, R: Rng + Clone> GmmValidParams<F, R> {
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn covariance_type(&self) -> &GmmCovarType {
        &self.covar_type
    }

    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    pub fn reg_covariance(&self) -> F {
        self.reg_covar
    }

    pub fn n_runs(&self) -> u64 {
        self.n_runs
    }

    pub fn max_n_iterations(&self) -> u64 {
        self.max_n_iter
    }

    pub fn init_method(&self) -> &GmmInitMethod {
        &self.init_method
    }

    pub fn rng(&self) -> R {
        self.rng.clone()
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug)]
/// The set of hyperparameters that can be specified for the execution of
/// the [GMM algorithm](struct.GaussianMixtureModel.html).
pub struct GmmParams<F: Float, R: Rng>(GmmValidParams<F, R>);

impl<F: Float> GmmParams<F, Isaac64Rng> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(n_clusters: usize) -> GmmParams<F, Isaac64Rng> {
        Self::new_with_rng(n_clusters, Isaac64Rng::seed_from_u64(42))
    }
}

impl<F: Float, R: Rng + Clone> GmmParams<F, R> {
    pub fn new_with_rng(n_clusters: usize, rng: R) -> GmmParams<F, R> {
        Self(GmmValidParams {
            n_clusters,
            covar_type: GmmCovarType::Full,
            tolerance: F::cast(1e-3),
            reg_covar: F::cast(1e-6),
            n_runs: 1,
            max_n_iter: 100,
            init_method: GmmInitMethod::KMeans,
            rng,
        })
    }

    /// Set the covariance type.
    pub fn covariance_type(mut self, covar_type: GmmCovarType) -> Self {
        self.0.covar_type = covar_type;
        self
    }

    /// Set the convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.tolerance = tolerance;
        self
    }

    /// Non-negative regularization added to the diagonal of covariance.
    /// Allows to assure that the covariance matrices are all positive.
    pub fn reg_covariance(mut self, reg_covar: F) -> Self {
        self.0.reg_covar = reg_covar;
        self
    }

    /// Set the number of initializations to perform. The best results are kept.
    pub fn n_runs(mut self, n_runs: u64) -> Self {
        self.0.n_runs = n_runs;
        self
    }

    /// Set the number of EM iterations to perform.
    pub fn max_n_iterations(mut self, max_n_iter: u64) -> Self {
        self.0.max_n_iter = max_n_iter;
        self
    }

    /// Set the method used to initialize the weights, the means and the precisions.
    pub fn init_method(mut self, init_method: GmmInitMethod) -> Self {
        self.0.init_method = init_method;
        self
    }

    pub fn with_rng<R2: Rng + Clone>(self, rng: R2) -> GmmParams<F, R2> {
        GmmParams(GmmValidParams {
            n_clusters: self.0.n_clusters,
            covar_type: self.0.covar_type,
            tolerance: self.0.tolerance,
            reg_covar: self.0.reg_covar,
            n_runs: self.0.n_runs,
            max_n_iter: self.0.max_n_iter,
            init_method: self.0.init_method,
            rng,
        })
    }
}

impl<F: Float, R: Rng> ParamGuard for GmmParams<F, R> {
    type Checked = GmmValidParams<F, R>;
    type Error = GmmError;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.n_clusters == 0 {
            Err(GmmError::InvalidValue(
                "`n_clusters` cannot be 0!".to_string(),
            ))
        } else if self.0.tolerance <= F::zero() {
            Err(GmmError::InvalidValue(
                "`tolerance` must be greater than 0!".to_string(),
            ))
        } else if self.0.reg_covar < F::zero() {
            Err(GmmError::InvalidValue(
                "`reg_covar` must be positive!".to_string(),
            ))
        } else if self.0.n_runs == 0 {
            Err(GmmError::InvalidValue("`n_runs` cannot be 0!".to_string()))
        } else if self.0.max_n_iter == 0 {
            Err(GmmError::InvalidValue(
                "`max_n_iterations` cannot be 0!".to_string(),
            ))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
