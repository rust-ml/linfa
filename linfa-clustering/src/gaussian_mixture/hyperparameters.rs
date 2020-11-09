use linfa::Float;
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
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
pub struct GmmHyperParams<F: Float, R: Rng> {
    n_clusters: usize,
    covar_type: GmmCovarType,
    tolerance: F,
    reg_covar: F,
    n_init: u64,
    max_n_iter: u64,
    init_method: GmmInitMethod,
    rng: R,
}

/// Helper struct used to construct a set of valid hyperparameters for
/// the execution of the [GMM algorithm](struct.GaussianMixtureModel.html).
pub struct GmmHyperParamsBuilder<F: Float, R: Rng> {
    n_clusters: usize,
    covar_type: GmmCovarType,
    tolerance: F,
    reg_covar: F,
    n_init: u64,
    max_n_iter: u64,
    init_method: GmmInitMethod,
    rng: R,
}

impl<F: Float, R: Rng + Clone> GmmHyperParamsBuilder<F, R> {
    /// Set the covariance type.
    pub fn covariance_type(mut self, covar_type: GmmCovarType) -> Self {
        self.covar_type = covar_type;
        self
    }

    /// Set the convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Non-negative regularization added to the diagonal of covariance.
    /// Allows to assure that the covariance matrices are all positive.
    pub fn reg_covariance(mut self, reg_covar: F) -> Self {
        self.reg_covar = reg_covar;
        self
    }

    /// Set the number of initializations to perform. The best results are kept.
    pub fn n_init(mut self, n_init: u64) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the number of EM iterations to perform.
    pub fn max_n_iterations(mut self, max_n_iter: u64) -> Self {
        self.max_n_iter = max_n_iter;
        self
    }

    /// Set the method used to initialize the weights, the means and the precisions.
    pub fn init_method(mut self, init_method: GmmInitMethod) -> Self {
        self.init_method = init_method;
        self
    }

    pub fn build(self) -> GmmHyperParams<F, R> {
        GmmHyperParams::build(
            self.n_clusters,
            self.covar_type,
            self.tolerance,
            self.reg_covar,
            self.n_init,
            self.max_n_iter,
            self.init_method,
            self.rng,
        )
    }
}

impl<F: Float> GmmHyperParams<F, Isaac64Rng> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(n_clusters: usize) -> GmmHyperParamsBuilder<F, Isaac64Rng> {
        Self::new_with_rng(n_clusters, Isaac64Rng::seed_from_u64(42))
    }
}

impl<F: Float, R: Rng + Clone> GmmHyperParams<F, R> {
    pub fn new_with_rng(n_clusters: usize, rng: R) -> GmmHyperParamsBuilder<F, R> {
        GmmHyperParamsBuilder {
            n_clusters,
            covar_type: GmmCovarType::Full,
            tolerance: F::from(1e-3).unwrap(),
            reg_covar: F::from(1e-6).unwrap(),
            n_init: 20,
            max_n_iter: 100,
            init_method: GmmInitMethod::KMeans,
            rng,
        }
    }

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

    pub fn n_init(&self) -> u64 {
        self.n_init
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

    #[allow(clippy::too_many_arguments)]
    pub fn build(
        n_clusters: usize,
        covar_type: GmmCovarType,
        tolerance: F,
        reg_covar: F,
        n_init: u64,
        max_n_iter: u64,
        init_method: GmmInitMethod,
        rng: R,
    ) -> Self {
        if n_clusters == 0 {
            panic!("`n_clusters` cannot be 0!");
        }
        if tolerance <= F::zero() {
            panic!("`tolerance` must be greater than 0!");
        }
        if reg_covar <= F::zero() {
            panic!("`reg_covar` must be greater than 0!");
        }
        if n_init == 0 {
            panic!("`n_init` cannot be 0!");
        }
        if max_n_iter == 0 {
            panic!("`max_n_iterations` cannot be 0!");
        }

        GmmHyperParams {
            n_clusters,
            covar_type,
            tolerance,
            reg_covar,
            n_init,
            max_n_iter,
            init_method,
            rng,
        }
    }
}
