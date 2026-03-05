//! L2Boosting (forward stagewise additive modelling with squared-error loss)
//! for the linfa ML framework.
//!
//! This module provides [`ResidualChain`], which fits models sequentially on
//! residuals. Chain as many stages as you like via [`Stagewise`]:
//!
//! 1. Fit `base` on `(X, Y)`
//! 2. Compute residuals: `R = Y - base.predict(X)`
//! 3. Fit `corrector` on `(X, R)`
//! 4. Repeat for any further correctors stacked on top
//!
//! When predicting, all stages' outputs are summed.
//!
//! This is the special case of FSAM (Friedman 2001) where the loss is squared
//! error. Shrinkage (learning rate ν ∈ (0, 1]) can be set per corrector via
//! [`Shrunk::with_shrinkage`]; the default is ν = 1 (no scaling).
//!
//! # References
//!
//! - J. H. Friedman (2001). "Greedy function approximation: A gradient boosting machine."
//!   <https://doi.org/10.1214/aos/1013203451>
//!
//! # Examples
//!
//! ## Linear + linear
//!
//! Two `linfa_linear::LinearRegression` models stacked: the corrector fits
//! the residuals left by the base.
//!
//! ```
//! use linfa::traits::{Fit, Predict};
//! use linfa::DatasetBase;
//! use linfa_linear::LinearRegression;
//! use linfa::composing::residual_chain::{ResidualChain, Stagewise};
//! use ndarray::{array, Array2};
//!
//! // y = 2x: perfectly linear, so the corrector should see zero residuals.
//! let x = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
//! let y = array![0., 2., 4., 6., 8.];
//! let dataset = DatasetBase::new(x.clone(), y);
//!
//! let fitted = LinearRegression::default()
//!     .stack_with(LinearRegression::default().shrink_by(1.0))
//!     .fit(&dataset)
//!     .unwrap();
//!
//! let _preds = fitted.predict(&x);
//! ```
//!
//! ## The second model learns nothing when the first fits perfectly
//!
//! If the first model already captures the data exactly, the residuals are all
//! zero and the second model has nothing to learn — its parameters come out
//! at (or very near) zero.
//!
//! ```
//! use linfa::traits::{Fit, Predict};
//! use linfa::DatasetBase;
//! use linfa_linear::LinearRegression;
//! use linfa::composing::residual_chain::Stagewise;
//! use ndarray::{array, Array2};
//!
//! // y = 2x: one linear model is enough to fit this perfectly.
//! let x = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
//! let y = array![0., 2., 4., 6., 8.];
//! let dataset = DatasetBase::new(x.clone(), y);
//!
//! let fitted = LinearRegression::default()
//!     .stack_with(LinearRegression::default().shrink_by(1.0))
//!     .fit(&dataset)
//!     .unwrap();
//!
//! // The corrector trained on zero residuals — nothing left to correct.
//! assert!(fitted.corrector().model().params().iter().all(|&c: &f64| c.abs() < 1e-10));
//! assert!(fitted.corrector().model().intercept().abs() < 1e-10);
//! ```
//!
//! ## Chained SVMs and linear regression
//!
//! A linear-kernel `linfa_svm::Svm` captures the overall trend; two
//! Gaussian-kernel SVMs and a `linfa_linear::LinearRegression` then fit
//! successive residuals in a four-model chain.
//!
//! ```
//! use linfa::traits::{Fit, Predict};
//! use linfa::DatasetBase;
//! use linfa_linear::LinearRegression;
//! use linfa::composing::residual_chain::{ResidualChain, Stagewise};
//! use linfa_svm::Svm;
//! use ndarray::Array;
//!
//! // y = sin(x): the linear SVM captures the slope; the RBF SVM captures
//! // the curvature left in the residuals.
//! let x = Array::linspace(0f64, 6., 20)
//!     .into_shape_with_order((20, 1))
//!     .unwrap();
//! let y = x.column(0).mapv(f64::sin);
//! let dataset = DatasetBase::new(x.clone(), y);
//!
//! let fitted = Svm::<f64, f64>::params()
//!     .c_svr(1., None)
//!     .linear_kernel()
//!     .stack_with(
//!         Svm::<f64, f64>::params()
//!             .c_svr(10., Some(0.1))
//!             .gaussian_kernel(1.)
//!             .shrink_by(1.0),
//!     )
//!     .stack_with(LinearRegression::default().shrink_by(1.0))
//!     .stack_with(
//!         Svm::<f64, f64>::params()
//!             .c_svr(10., Some(0.1))
//!             .gaussian_kernel(3.)
//!             .shrink_by(1.0),
//!     )
//!     .fit(&dataset)
//!     .unwrap();
//!
//! let _preds = fitted.predict(&x);
//! ```

use crate::dataset::{AsTargets, DatasetBase, Records};
use crate::param_guard::ParamGuard;
use crate::traits::{Fit, Predict, PredictInplace};
use crate::Float;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2, RawDataClone};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::ops::{AddAssign, Mul};

type Arr2<D> = ArrayBase<D, Ix2>;

/// Error returned by [`ResidualChain::fit`].
///
/// Wraps the error from whichever of the two model fits failed, keeping them
/// distinguishable without requiring both models to share the same error type.
#[derive(Debug, thiserror::Error)]
pub enum ResidualChainError<E1, E2> {
    #[error("base model: {0}")]
    Base(E1),
    #[error("corrector: {0}")]
    Corrector(E2),
    // Satisfies the `Fit` trait's `E: From<linfa::error::Error>` bound.
    #[error(transparent)]
    BaseCrate(#[from] crate::Error),
}

/// A pair of [`Fit`] params that fits sequentially on residuals.
///
/// `base` is fit on the original targets; `corrector` (a [`Shrunk`] model) is
/// fit on the residuals left by `base` and scaled by its shrinkage factor ν.
/// Prediction sums `base` and the scaled corrector output.
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct ResidualChain<B, C, F: Float> {
    base: B,
    corrector: Shrunk<C, F>,
}

impl<B, C, F: Float> ResidualChain<B, C, F> {
    pub fn base(&self) -> &B {
        &self.base
    }
    pub fn corrector(&self) -> &Shrunk<C, F> {
        &self.corrector
    }
}

/// Extension trait that adds residual-chain composition methods to any type.
///
/// Blanket-implemented for all `Sized` types, so any model params type gains
/// these methods automatically:
///
/// - [`stack_with`](Stagewise::stack_with): compose `self` (as the base) with
///   a [`Shrunk`] corrector that will be trained on the residuals left by
///   `self`. Returns a [`ResidualChainParams`] whose `.fit()` runs both stages.
///   Calls can be chained to build arbitrarily deep sequences.
/// - [`shrink_by`](Stagewise::shrink_by): wrap `self` in a [`Shrunk`] with the
///   given learning rate ν, making it ready to pass as the `corrector` argument
///   to [`Stagewise::stack_with`].
///
/// # Example
///
/// ```
/// use linfa::traits::Fit;
/// use linfa::DatasetBase;
/// use linfa_linear::LinearRegression;
/// use linfa::composing::residual_chain::Stagewise;
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
/// let y = array![0., 2., 4., 6., 8.];
/// let dataset = DatasetBase::new(x.clone(), y);
///
/// let fitted = LinearRegression::default()
///     .stack_with(LinearRegression::default().shrink_by(1.0))
///     .fit(&dataset)
///     .unwrap();
/// ```
pub trait Stagewise: Sized {
    /// Compose `self` (as the base model) with `corrector`, which will be
    /// trained on the residuals left by `self`. Further stages can be appended
    /// by calling `.stack_with(...)` on the returned [`ResidualChainParams`].
    fn stack_with<C, F: Float>(self, corrector: Shrunk<C, F>) -> ResidualChainParams<Self, C, F>;
    /// Wrap `self` in a [`Shrunk`] with learning rate `shrinkage` ∈ (0, 1],
    /// making it ready to pass as the `corrector` argument to [`Stagewise::stack_with`].
    ///
    /// The bound `Self: Fit<Array2<F>, Array1<F>, E>` ensures at compile time
    /// that the model's element type matches the shrinkage type `F`.
    fn shrink_by<F: Float, E>(self, shrinkage: F) -> Shrunk<Self, F>
    where
        Self: Fit<Array2<F>, Array1<F>, E>,
        E: std::error::Error + From<crate::error::Error>;
}

impl<B> Stagewise for B {
    fn stack_with<C, F: Float>(self, corrector: Shrunk<C, F>) -> ResidualChainParams<B, C, F> {
        ResidualChainParams(ResidualChain {
            base: self,
            corrector,
        })
    }
    fn shrink_by<F: Float, E>(self, shrinkage: F) -> Shrunk<Self, F>
    where
        Self: Fit<Array2<F>, Array1<F>, E>,
        E: std::error::Error + From<crate::error::Error>,
    {
        Shrunk {
            model: self,
            shrinkage,
        }
    }
}

impl<F1, F2, F: Float, D: Data<Elem = F> + RawDataClone, T, E1, E2>
    Fit<Arr2<D>, T, ResidualChainError<E1, E2>> for ResidualChain<F1, F2, F>
where
    Arr2<D>: Records,
    F1: Fit<Arr2<D>, T, E1>,
    for<'a> F1::Object: Predict<&'a Arr2<D>, Array1<F>>,
    F2: Fit<Arr2<D>, Array1<F>, E2>,
    T: AsTargets<Elem = F, Ix = Ix1>,
    E1: std::error::Error + From<crate::error::Error>,
    E2: std::error::Error + From<crate::error::Error>,
{
    type Object = ResidualChain<F1::Object, F2::Object, F>;

    fn fit(
        &self,
        dataset: &DatasetBase<Arr2<D>, T>,
    ) -> Result<Self::Object, ResidualChainError<E1, E2>> {
        let base = self.base.fit(dataset).map_err(ResidualChainError::Base)?;

        let y_pred = base.predict(dataset.records());
        let residuals = &dataset.targets().as_targets() - &y_pred;

        let residual_dataset = DatasetBase::new(dataset.records().clone(), residuals);
        let corrector_model = self
            .corrector
            .model
            .fit(&residual_dataset)
            .map_err(ResidualChainError::Corrector)?;

        Ok(ResidualChain {
            base,
            corrector: Shrunk {
                model: corrector_model,
                shrinkage: self.corrector.shrinkage,
            },
        })
    }
}

impl<R1, R2, F: Float, D: Data<Elem = F>> PredictInplace<Arr2<D>, Array1<F>>
    for ResidualChain<R1, R2, F>
where
    R1: PredictInplace<Arr2<D>, Array1<F>>,
    R2: PredictInplace<Arr2<D>, Array1<F>>,
{
    fn predict_inplace<'a>(&'a self, x: &'a Arr2<D>, y: &mut Array1<F>) {
        self.base.predict_inplace(x, y);
        y.add_assign(
            &self
                .corrector
                .model
                .predict(x)
                .mul(self.corrector.shrinkage),
        );
    }

    fn default_target(&self, x: &Arr2<D>) -> Array1<F> {
        Array1::zeros(x.nrows())
    }
}

/// A model (params or fitted) paired with a shrinkage factor ν ∈ (0, 1].
///
/// Used in two roles:
/// - **Before fitting**: `Shrunk<C, F>` wraps corrector params `C`; created by
///   [`Stagewise::shrink_by`] and stored in [`ResidualChain`] / [`ResidualChainParams`].
/// - **After fitting**: `Shrunk<C::Object, F>` wraps the fitted corrector model;
///   prediction scales the corrector's output by ν before summing with the base.
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct Shrunk<M, F: Float> {
    model: M,
    shrinkage: F,
}

impl<M, F: Float> Shrunk<M, F> {
    pub fn model(&self) -> &M {
        &self.model
    }
    pub fn shrinkage(&self) -> F {
        self.shrinkage
    }
    /// Set the shrinkage factor. Validation happens when the containing
    /// [`ResidualChainParams`] is checked via [`ParamGuard`].
    pub fn with_shrinkage(mut self, shrinkage: F) -> Self {
        self.shrinkage = shrinkage;
        self
    }
}

/// Unvalidated [`ResidualChain`] parameters returned by [`Stagewise::stack_with`].
///
/// Call `.fit()` to validate and fit in one step — the [`ParamGuard`] blanket
/// impl runs `check_ref` first, which verifies that the outermost corrector's
/// shrinkage factor is in (0, 1]. Inner chains validate lazily when their own
/// `.fit()` is called. You can also call `.check()` / `.check_unwrap()` to
/// validate explicitly.
///
/// To set an explicit shrinkage factor on the corrector use
/// [`Shrunk::with_shrinkage`]:
///
/// ```
/// use linfa::traits::{Fit, Predict};
/// use linfa::DatasetBase;
/// use linfa_linear::LinearRegression;
/// use linfa::composing::residual_chain::{Shrunk, Stagewise};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
/// let y = array![0., 2., 4., 6., 8.];
/// let dataset = DatasetBase::new(x.clone(), y);
///
/// // The corrector's contribution is scaled by 0.1.
/// let fitted = LinearRegression::default()
///     .stack_with(LinearRegression::default().shrink_by(0.1))
///     .fit(&dataset)
///     .unwrap();
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct ResidualChainParams<B, C, F: Float>(ResidualChain<B, C, F>);

impl<B, C, F: Float> ParamGuard for ResidualChainParams<B, C, F> {
    type Checked = ResidualChain<B, C, F>;
    type Error = crate::Error;

    fn check_ref(&self) -> Result<&ResidualChain<B, C, F>, crate::Error> {
        let v = self.0.corrector.shrinkage;
        if v > F::zero() && v <= F::one() {
            Ok(&self.0)
        } else {
            Err(crate::Error::Parameters(format!(
                "shrinkage must be in (0, 1], got {v}"
            )))
        }
    }

    fn check(self) -> Result<ResidualChain<B, C, F>, crate::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error as LinfaError;
    use crate::DatasetBase;
    use ndarray::{array, Array1, Array2};

    #[derive(thiserror::Error, Debug)]
    #[error("dummy error")]
    struct DummyError(#[from] LinfaError);

    // Params that fits by recording the mean of the targets.
    struct MeanParams;

    // Model that predicts the mean it saw during fit.
    struct MeanModel(f64);

    impl Fit<Array2<f64>, Array1<f64>, DummyError> for MeanParams {
        type Object = MeanModel;
        fn fit(
            &self,
            dataset: &DatasetBase<Array2<f64>, Array1<f64>>,
        ) -> Result<MeanModel, DummyError> {
            let mean = dataset.targets().iter().sum::<f64>() / dataset.targets().len() as f64;
            Ok(MeanModel(mean))
        }
    }

    impl PredictInplace<Array2<f64>, Array1<f64>> for MeanModel {
        fn predict_inplace(&self, x: &Array2<f64>, y: &mut Array1<f64>) {
            y.assign(&Array1::from_elem(x.nrows(), self.0));
        }
        fn default_target(&self, x: &Array2<f64>) -> Array1<f64> {
            Array1::zeros(x.nrows())
        }
    }

    #[test]
    fn corrector_is_fit_on_residuals() {
        // targets = [1, 3]. base sees mean=2, predicts 2 for all.
        // residuals = [1-2, 3-2] = [-1, 1]. corrector sees mean=0.
        let model = MeanParams.stack_with(MeanParams.shrink_by(1.0));
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![1.0, 3.0]);
        let fitted = model.fit(&dataset).unwrap();

        assert_eq!(fitted.base().0, 2.0); // mean of [1, 3]
        assert_eq!(fitted.corrector().model().0, 0.0); // mean of residuals [-1, 1]
    }

    #[test]
    fn predict_sums_both_models() {
        // base predicts 2.0, corrector predicts 0.0 → sum = 2.0
        let model = MeanParams.stack_with(MeanParams.shrink_by(1.0));
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![1.0, 3.0]);
        let fitted = model.fit(&dataset).unwrap();

        let records = array![[0.0_f64], [1.0]];
        let predictions = fitted.predict(&records);
        assert_eq!(predictions, array![2.0, 2.0]);
    }

    #[test]
    fn predict_recovers_targets_when_residuals_fit_perfectly() {
        // If the corrector perfectly fits the residuals, the combined prediction = original targets.
        struct FixedParams(f64);
        struct FixedModel(f64);

        impl Fit<Array2<f64>, Array1<f64>, DummyError> for FixedParams {
            type Object = FixedModel;
            fn fit(
                &self,
                _dataset: &DatasetBase<Array2<f64>, Array1<f64>>,
            ) -> Result<FixedModel, DummyError> {
                Ok(FixedModel(self.0))
            }
        }

        impl PredictInplace<Array2<f64>, Array1<f64>> for FixedModel {
            fn predict_inplace(&self, x: &Array2<f64>, y: &mut Array1<f64>) {
                y.assign(&Array1::from_elem(x.nrows(), self.0));
            }
            fn default_target(&self, x: &Array2<f64>) -> Array1<f64> {
                Array1::zeros(x.nrows())
            }
        }

        // base predicts 3.0, corrector predicts 1.0 → sum = 4.0
        let model = FixedParams(3.0)
            .stack_with(FixedParams(1.0).shrink_by(1.0))
            .stack_with(FixedParams(0.0).shrink_by(1.0));
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![4.0, 4.0]);
        let fitted = model.fit(&dataset).unwrap();

        let predictions = fitted.predict(&array![[0.0_f64], [1.0]]);
        assert_eq!(predictions, array![4.0, 4.0]);
    }

    #[test]
    fn deep_chain_accessors() {
        let model = MeanParams
            .stack_with(MeanParams.shrink_by(1.0))
            .stack_with(MeanParams.shrink_by(1.0))
            .stack_with(MeanParams.shrink_by(1.0));
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![1.0, 3.0]);
        let fitted = model.fit(&dataset).unwrap();

        assert_eq!(fitted.base().base().base().0, 2.0); // params trained on original targets
    }

    #[test]
    fn shrinkage_scales_corrector_prediction() {
        // base predicts mean=2.0, corrector predicts mean=0.0 (residuals [-1,1]).
        // With shrinkage=0.5, corrector contributes 0.5*0.0 = 0.0 → total = 2.0.
        let model = MeanParams.stack_with(MeanParams.shrink_by(0.5));
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![1.0, 3.0]);
        let fitted = model.fit(&dataset).unwrap();

        let preds = fitted.predict(&array![[0.0_f64], [1.0]]);
        assert_eq!(preds, array![2.0, 2.0]);
        assert_eq!(fitted.corrector().shrinkage(), 0.5);
    }

    #[test]
    fn shrinkage_corrector_sees_scaled_residuals() {
        // base predicts 3.0 always. targets = [4.0, 4.0].
        // residuals = [1.0, 1.0]. corrector (mean) sees mean=1.0.
        // With shrinkage=0.5: prediction = 3.0 + 0.5*1.0 = 3.5.
        struct FixedParams(f64);
        struct FixedModel(f64);

        impl Fit<Array2<f64>, Array1<f64>, DummyError> for FixedParams {
            type Object = FixedModel;
            fn fit(
                &self,
                _dataset: &DatasetBase<Array2<f64>, Array1<f64>>,
            ) -> Result<FixedModel, DummyError> {
                Ok(FixedModel(self.0))
            }
        }

        impl PredictInplace<Array2<f64>, Array1<f64>> for FixedModel {
            fn predict_inplace(&self, x: &Array2<f64>, y: &mut Array1<f64>) {
                y.assign(&Array1::from_elem(x.nrows(), self.0));
            }
            fn default_target(&self, x: &Array2<f64>) -> Array1<f64> {
                Array1::zeros(x.nrows())
            }
        }

        let model = FixedParams(3.0).stack_with(MeanParams.shrink_by(0.5));
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![4.0, 4.0]);
        let fitted = model.fit(&dataset).unwrap();

        let preds = fitted.predict(&array![[0.0_f64], [1.0]]);
        // corrector saw residuals [1.0, 1.0], mean=1.0, shrunk by 0.5 → 0.5
        assert!((preds[0] - 3.5_f64).abs() < 1e-10);
    }

    #[test]
    fn shrinkage_invalid_value_returns_error() {
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![1.0, 3.0]);
        let model = MeanParams.stack_with(MeanParams.shrink_by(0.0));
        assert!(model.fit(&dataset).is_err());

        let model = MeanParams.stack_with(MeanParams.shrink_by(1.5));
        assert!(model.fit(&dataset).is_err());
    }
}
