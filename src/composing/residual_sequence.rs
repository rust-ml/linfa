//! Residual sequence model composition for the linfa ML framework.
//!
//! This crate provides [`ResidualSequence`], which fits models sequentially on
//! the residuals of the previous one. Chain as many as you like via [`StackWith`]:
//!
//! 1. Fit `first` on `(X, Y)`
//! 2. Compute residuals: `R = Y - first.predict(X)`
//! 3. Fit `second` on `(X, R)`
//! 4. Repeat for any further models stacked on top
//!
//! When predicting, all models' outputs are summed.
//!
//! This is the foundation of boosting / residual stacking.
//!
//! # Examples
//!
//! ## Linear + linear
//!
//! Two `linfa_linear::LinearRegression` models stacked: the second fits the
//! residuals left by the first.
//!
//! ```
//! use linfa::traits::{Fit, Predict};
//! use linfa::DatasetBase;
//! use linfa_linear::LinearRegression;
//! use linfa::composing::residual_sequence::{ResidualSequence, StackWith};
//! use ndarray::{array, Array2};
//!
//! // y = 2x: perfectly linear, so the second model should see zero residuals.
//! let x = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
//! let y = array![0., 2., 4., 6., 8.];
//! let dataset = DatasetBase::new(x.clone(), y);
//!
//! let fitted = LinearRegression::default()
//!     .stack_with(LinearRegression::default())
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
//! use linfa::composing::residual_sequence::StackWith;
//! use ndarray::{array, Array2};
//!
//! // y = 2x: one linear model is enough to fit this perfectly.
//! let x = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
//! let y = array![0., 2., 4., 6., 8.];
//! let dataset = DatasetBase::new(x.clone(), y);
//!
//! let fitted = LinearRegression::default()
//!     .stack_with(LinearRegression::default())
//!     .fit(&dataset)
//!     .unwrap();
//!
//! // The second model trained on zero residuals — nothing left to correct.
//! assert!(fitted.second.params().iter().all(|&c: &f64| c.abs() < 1e-10));
//! assert!(fitted.second.intercept().abs() < 1e-10);
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
//! use linfa::composing::residual_sequence::{ResidualSequence, StackWith};
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
//!             .gaussian_kernel(1.),
//!     )
//!     .stack_with(LinearRegression::default())
//!     .stack_with(
//!         Svm::<f64, f64>::params()
//!             .c_svr(10., Some(0.1))
//!             .gaussian_kernel(3.),
//!     )
//!     .fit(&dataset)
//!     .unwrap();
//!
//! let _preds = fitted.predict(&x);
//! ```

use crate::dataset::{AsTargets, DatasetBase, Records};
use crate::traits::{Fit, Predict, PredictInplace};
use crate::{Float, ParamGuard};
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2, RawDataClone};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::ops::AddAssign;

type Arr2<D> = ArrayBase<D, Ix2>;

/// Error returned by [`ResidualSequence::fit`].
///
/// Wraps the error from whichever of the two model fits failed, keeping them
/// distinguishable without requiring both models to share the same error type.
#[derive(Debug, thiserror::Error)]
pub enum ResidualSequenceError<E1, E2> {
    #[error("first model: {0}")]
    First(E1),
    #[error("second model: {0}")]
    Second(E2),
    // Satisfies the `Fit` trait's `E: From<linfa::error::Error>` bound.
    #[error(transparent)]
    BaseCrate(#[from] crate::Error),
}

/// Error returned when checking [`ResidualSequence`] hyperparameters.
///
/// Wraps the validation error from whichever sub-model's parameter check failed.
#[derive(Debug, thiserror::Error)]
pub enum ResidualParamError<E1, E2> {
    #[error("first model params: {0}")]
    First(E1),
    #[error("second model params: {0}")]
    Second(E2),
}

/// Fits two models sequentially on the residuals of the first.
///
/// `first` is fit on the original dataset. `second` is fit on the residuals
/// `Y - first.predict(X)`. See the [module docs](self) for details.
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct ResidualSequence<F1, F2> {
    first: F1,
    second: F2,
}

impl<F1, F2> ResidualSequence<F1, F2> {
    pub fn first(&self) -> &F1 {
        &self.first
    }
    pub fn second(&self) -> &F2 {
        &self.second
    }
}

/// Extension trait that lets any model params type be composed into a [`ResidualSequence`].
///
/// # Example
///
/// ```
/// use linfa::traits::Fit;
/// use linfa::DatasetBase;
/// use linfa_linear::LinearRegression;
/// use linfa::composing::residual_sequence::StackWith;
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
/// let y = array![0., 2., 4., 6., 8.];
/// let dataset = DatasetBase::new(x.clone(), y);
///
/// let fitted = LinearRegression::default()
///     .stack_with(LinearRegression::default())
///     .fit(&dataset)
///     .unwrap();
/// ```
pub trait StackWith: Sized {
    /// Wrap `self` and `second` into a [`ResidualSequence`] that will fit
    /// `second` on the residuals left by `self`. Calls can be chained to add
    /// further stages.
    fn stack_with<F2>(self, second: F2) -> ResidualSequence<Self, F2>;
}

impl<F1> StackWith for F1 {
    fn stack_with<F2>(self, second: F2) -> ResidualSequence<F1, F2> {
        ResidualSequence {
            first: self,
            second,
        }
    }
}

impl<F1, F2> ParamGuard for ResidualSequence<F1, F2>
where
    F1: ParamGuard<Checked = F1>,
    F2: ParamGuard<Checked = F2>,
{
    type Checked = Self;
    type Error = ResidualParamError<F1::Error, F2::Error>;

    /// Validates both sub-model hyperparameters.
    ///
    /// Returns a reference to `self` if both pass, or the first error encountered.
    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        self.first.check_ref().map_err(ResidualParamError::First)?;
        self.second.check_ref().map_err(ResidualParamError::Second)?;
        Ok(self)
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self)
    }
}

/// Two fitted models produced by [`ResidualSequence::fit`].
///
/// Predicts by summing both models' outputs: `first.predict(X) + second.predict(X)`.
#[derive(Debug, Clone)]
pub struct FittedResidualSequence<R1, R2> {
    pub first: R1,
    pub second: R2,
}

impl<F1, F2, F: Float, D: Data<Elem = F> + RawDataClone, T, E1, E2>
    Fit<Arr2<D>, T, ResidualSequenceError<E1, E2>> for ResidualSequence<F1, F2>
where
    Arr2<D>: Records,
    F1: Fit<Arr2<D>, T, E1>,
    for<'a> F1::Object: Predict<&'a Arr2<D>, Array1<F>>,
    F2: Fit<Arr2<D>, Array1<F>, E2>,
    T: AsTargets<Elem = F, Ix = Ix1>,
    E1: std::error::Error + From<crate::error::Error>,
    E2: std::error::Error + From<crate::error::Error>,
{
    type Object = FittedResidualSequence<F1::Object, F2::Object>;

    fn fit(
        &self,
        dataset: &DatasetBase<Arr2<D>, T>,
    ) -> Result<Self::Object, ResidualSequenceError<E1, E2>> {
        let first = self
            .first
            .fit(dataset)
            .map_err(ResidualSequenceError::First)?;

        let y_pred = first.predict(dataset.records());
        let residuals = &dataset.targets().as_targets() - &y_pred;

        let residual_dataset = DatasetBase::new(dataset.records().clone(), residuals);
        let second = self
            .second
            .fit(&residual_dataset)
            .map_err(ResidualSequenceError::Second)?;

        Ok(FittedResidualSequence { first, second })
    }
}

impl<R1, R2, F: Float, D: Data<Elem = F>> PredictInplace<Arr2<D>, Array1<F>>
    for FittedResidualSequence<R1, R2>
where
    for<'a> R1: Predict<&'a Arr2<D>, Array1<F>>,
    for<'a> R2: Predict<&'a Arr2<D>, Array1<F>>,
{
    fn predict_inplace<'a>(&'a self, x: &'a Arr2<D>, y: &mut Array1<F>) {
        y.assign(&self.first.predict(x));
        y.add_assign(&self.second.predict(x));
    }

    fn default_target(&self, x: &Arr2<D>) -> Array1<F> {
        Array1::zeros(x.nrows())
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

    // --- ParamGuard helpers ---

    // Error used by test ParamGuard stubs.
    #[derive(thiserror::Error, Debug, PartialEq)]
    #[error("invalid params: {0}")]
    struct ParamErr(String);

    // Always-valid params stub.
    #[derive(Debug)]
    struct OkParams;

    impl ParamGuard for OkParams {
        type Checked = Self;
        type Error = ParamErr;
        fn check_ref(&self) -> Result<&Self, ParamErr> {
            Ok(self)
        }
        fn check(self) -> Result<Self, ParamErr> {
            Ok(self)
        }
    }

    // Always-invalid params stub.
    #[derive(Debug)]
    struct BadParams(String);

    impl ParamGuard for BadParams {
        type Checked = Self;
        type Error = ParamErr;
        fn check_ref(&self) -> Result<&Self, ParamErr> {
            Err(ParamErr(self.0.clone()))
        }
        fn check(self) -> Result<Self, ParamErr> {
            Err(ParamErr(self.0))
        }
    }

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

    impl<'a> Predict<&'a Array2<f64>, Array1<f64>> for MeanModel {
        fn predict(&self, x: &'a Array2<f64>) -> Array1<f64> {
            Array1::from_elem(x.nrows(), self.0)
        }
    }

    #[test]
    fn second_is_fit_on_residuals() {
        // targets = [1, 3]. first sees mean=2, predicts 2 for all.
        // residuals = [1-2, 3-2] = [-1, 1]. second sees mean=0.
        let model = ResidualSequence {
            first: MeanParams,
            second: MeanParams,
        };
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![1.0, 3.0]);
        let fitted = model.fit(&dataset).unwrap();

        assert_eq!(fitted.first.0, 2.0); // mean of [1, 3]
        assert_eq!(fitted.second.0, 0.0); // mean of residuals [-1, 1]
    }

    #[test]
    fn predict_sums_both_models() {
        // first predicts 2.0, second predicts 0.0 → sum = 2.0
        let model = MeanParams.stack_with(MeanParams);
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![1.0, 3.0]);
        let fitted = model.fit(&dataset).unwrap();

        let records = array![[0.0_f64], [1.0]];
        let predictions = fitted.predict(&records);
        assert_eq!(predictions, array![2.0, 2.0]);
    }

    #[test]
    fn predict_recovers_targets_when_residuals_fit_perfectly() {
        // If second perfectly fits the residuals, the combined prediction = original targets.
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

        impl<'a> Predict<&'a Array2<f64>, Array1<f64>> for FixedModel {
            fn predict(&self, x: &'a Array2<f64>) -> Array1<f64> {
                Array1::from_elem(x.nrows(), self.0)
            }
        }

        // first predicts 3.0, second predicts 1.0 → sum = 4.0
        let model = FixedParams(3.0)
            .stack_with(FixedParams(1.0))
            .stack_with(FixedParams(0.0));
        let dataset = DatasetBase::new(array![[0.0_f64], [1.0]], array![4.0, 4.0]);
        let fitted = model.fit(&dataset).unwrap();

        let predictions = fitted.predict(&array![[0.0_f64], [1.0]]);
        assert_eq!(predictions, array![4.0, 4.0]);
    }

    // --- ParamGuard tests ---

    #[test]
    fn param_guard_check_ref_succeeds_when_both_params_valid() {
        let seq = OkParams.stack_with(OkParams);
        assert!(seq.check_ref().is_ok());
    }

    #[test]
    fn param_guard_check_ref_fails_on_invalid_first() {
        let seq = BadParams("bad first".into()).stack_with(OkParams);
        let err = seq.check_ref().unwrap_err();
        assert!(matches!(err, ResidualParamError::First(ParamErr(_))));
    }

    #[test]
    fn param_guard_check_ref_fails_on_invalid_second() {
        let seq = OkParams.stack_with(BadParams("bad second".into()));
        let err = seq.check_ref().unwrap_err();
        assert!(matches!(err, ResidualParamError::Second(ParamErr(_))));
    }

    #[test]
    fn param_guard_check_succeeds_and_returns_self() {
        let seq = OkParams.stack_with(OkParams);
        assert!(seq.check().is_ok());
    }

    #[test]
    fn param_guard_check_fails_on_invalid_first() {
        let seq = BadParams("bad".into()).stack_with(OkParams);
        assert!(matches!(
            seq.check().unwrap_err(),
            ResidualParamError::First(_)
        ));
    }

    #[test]
    fn param_guard_check_fails_on_invalid_second() {
        let seq = OkParams.stack_with(BadParams("bad".into()));
        assert!(matches!(
            seq.check().unwrap_err(),
            ResidualParamError::Second(_)
        ));
    }
}
