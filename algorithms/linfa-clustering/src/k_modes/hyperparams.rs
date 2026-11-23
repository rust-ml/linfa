use crate::k_modes::init::KModesInit;
use crate::KModesParamsError;
use linfa::ParamGuard;
use ndarray_rand::rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
/// Hyperparameters that can be specified for
/// the [K-Modes algorithm](crate::KModes).
pub struct KModesValidParams<T, R: Rng> {
    /// Maximum number of iterations for a single run
    /// When max_n_iterations is exceeded we terminate training
    pub(crate) max_n_iterations: u64,
    /// Number of clusters to form
    pub(crate) n_clusters: usize,
    /// Number of times the algorithm will run with different seeds
    pub(crate) n_runs: usize,
    /// Centroid initialization methods
    pub(crate) init: KModesInit<T>,
    /// Enable verbose progress logging to stdout.
    pub(crate) verbose: bool,
    /// Random number generator.
    pub(crate) rng: R,
}

impl<T, R: Rng> KModesValidParams<T, R> {
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn max_n_iterations(&self) -> u64 {
        self.max_n_iterations
    }

    pub fn n_runs(&self) -> usize {
        self.n_runs
    }

    pub fn init_method(&self) -> &KModesInit<T> {
        &self.init
    }

    pub fn verbose(&self) -> bool {
        self.verbose
    }

    pub fn rng(&self) -> &R {
        &self.rng
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
/// Helper builder to configure hyperparameters for the [K-Modes algorithm](crate::KModes).
pub struct KModesParams<T, R: Rng>(KModesValidParams<T, R>);

impl<T> KModesParams<T, Xoshiro256Plus> {
    /// Create a new K-Modes parameter builder with default RNG (seed 42).
    /// Implemented with `new_with_rng` with default parameters:
    ///  * max_n_iterations: 100
    ///  * n_runs: 10
    ///  * init: KModesInit::Cao
    ///  * verbose: false
    pub fn new(n_clusters: usize) -> Self {
        Self::new_with_rng(n_clusters, Xoshiro256Plus::seed_from_u64(42))
    }
}

impl<T, R: Rng> KModesParams<T, R> {
    /// Create a new K-Modes parameter builder with a custom RNG.
    /// Defaults:
    ///  * max_n_iterations: 100
    ///  * n_runs: 10
    ///  * init: KModesInit::Cao
    ///  * verbose: false
    pub fn new_with_rng(n_clusters: usize, rng: R) -> Self {
        Self(KModesValidParams {
            n_clusters,
            max_n_iterations: 100,
            n_runs: 10,
            init: KModesInit::Cao,
            verbose: false,
            rng,
        })
    }

    /// Set the maximum number of iterations for a single run.
    pub fn max_n_iterations(mut self, max_n_iterations: u64) -> Self {
        self.0.max_n_iterations = max_n_iterations;
        self
    }

    /// Set the number of initialization runs (keeps the run with minimal cost/inertia).
    pub fn n_runs(mut self, n_runs: usize) -> Self {
        self.0.n_runs = n_runs;
        self
    }

    /// Set the centroid initialization method (`Cao`, `Huang`, `Random`, or `Precomputed`).
    pub fn init_method(mut self, init: KModesInit<T>) -> Self {
        self.0.init = init;
        self
    }

    /// Enable or disable verbose progress logging to stdout.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.0.verbose = verbose;
        self
    }

    /// Set a custom random number generator.
    pub fn with_rng<R2: Rng>(self, rng: R2) -> KModesParams<T, R2> {
        KModesParams(KModesValidParams {
            n_clusters: self.0.n_clusters,
            max_n_iterations: self.0.max_n_iterations,
            n_runs: self.0.n_runs,
            init: self.0.init,
            verbose: self.0.verbose,
            rng,
        })
    }
}

impl<T, R: Rng> ParamGuard for KModesParams<T, R> {
    type Checked = KModesValidParams<T, R>;
    type Error = KModesParamsError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.n_clusters == 0 {
            Err(KModesParamsError::NClusters)
        } else if self.0.max_n_iterations == 0 {
            Err(KModesParamsError::MaxIterations)
        } else if self.0.n_runs == 0 {
            Err(KModesParamsError::NRuns)
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::ParamGuard;
    use ndarray_rand::rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<KModesParamsError>();
        has_autotraits::<KModesParams<usize, Xoshiro256Plus>>();
        has_autotraits::<KModesValidParams<usize, Xoshiro256Plus>>();
    }

    #[test]
    fn invalid_params() {
        assert_eq!(
            KModesParams::<usize, _>::new(0).check(),
            Err(KModesParamsError::NClusters)
        );
        assert_eq!(
            KModesParams::<usize, _>::new(2).max_n_iterations(0).check(),
            Err(KModesParamsError::MaxIterations)
        );
        assert_eq!(
            KModesParams::<usize, _>::new(2).n_runs(0).check(),
            Err(KModesParamsError::NRuns)
        );
    }

    #[test]
    fn test_getters_and_builder_options() {
        let params = KModesParams::<&str, _>::new(3)
            .max_n_iterations(200)
            .n_runs(15)
            .init_method(KModesInit::Huang)
            .verbose(true);

        assert!(params.check_ref().is_ok());
        let valid = params.check().unwrap();
        assert_eq!(valid.n_clusters(), 3);
        assert_eq!(valid.max_n_iterations(), 200);
        assert_eq!(valid.n_runs(), 15);
        assert_eq!(valid.init_method(), &KModesInit::Huang);
        assert!(valid.verbose());
        let _ = valid.rng();
    }

    #[test]
    fn test_custom_rng() {
        let rng1 = Xoshiro256Plus::seed_from_u64(12345);
        let params1 = KModesParams::<usize, _>::new_with_rng(4, rng1);
        let valid1 = params1.check().unwrap();
        assert_eq!(valid1.n_clusters(), 4);

        let rng2 = Xoshiro256Plus::seed_from_u64(67890);
        let params2 = KModesParams::<usize, _>::new(2).with_rng(rng2);
        let valid2 = params2.check().unwrap();
        assert_eq!(valid2.n_clusters(), 2);
    }

    #[cfg(feature = "serde")]
    #[cfg_attr(
        feature = "serde",
        derive(Serialize, Deserialize),
        serde(crate = "serde_crate")
    )]
    #[derive(Clone, Debug, PartialEq)]
    struct DummyRng;

    #[cfg(feature = "serde")]
    impl ndarray_rand::rand::RngCore for DummyRng {
        fn next_u32(&mut self) -> u32 {
            0
        }
        fn next_u64(&mut self) -> u64 {
            0
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(0);
        }
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), ndarray_rand::rand::Error> {
            dest.fill(0);
            Ok(())
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_hyperparams() {
        use ndarray_rand::rand::RngCore;
        let params = KModesParams::<String, _>::new_with_rng(3, DummyRng)
            .max_n_iterations(50)
            .n_runs(5)
            .init_method(KModesInit::Cao)
            .verbose(false);
        let valid = params.check().unwrap();

        let serialized = serde_json::to_string(&valid).unwrap();
        let deserialized: KModesValidParams<String, DummyRng> =
            serde_json::from_str(&serialized).unwrap();

        let mut dummy = DummyRng;
        assert_eq!(dummy.next_u32(), 0);
        assert_eq!(dummy.next_u64(), 0);
        let mut buf = [0u8; 4];
        dummy.fill_bytes(&mut buf);
        assert!(dummy.try_fill_bytes(&mut buf).is_ok());

        assert_eq!(valid.n_clusters(), deserialized.n_clusters());
        assert_eq!(valid.max_n_iterations(), deserialized.max_n_iterations());
        assert_eq!(valid.n_runs(), deserialized.n_runs());
        assert_eq!(valid.init_method(), deserialized.init_method());
        assert_eq!(valid.verbose(), deserialized.verbose());
    }
}
