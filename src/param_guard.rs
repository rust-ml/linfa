use std::error::Error;

use crate::{
    prelude::Records,
    traits::{Fit, FitWith, Transformer},
};

/// A set of hyperparameters whose values have not been checked for validity. A reference to the
/// checked hyperparameters can only be obtained after checking has completed. If the
/// `Transformer`, `Fit`, or `FitWith` traits have been implemented on the checked
/// hyperparameters, they will also be implemented on the unchecked hyperparameters with the
/// checking step done automatically.
///
/// The hyperparameter validation done in `check_ref()` and `check()` should be identical.
pub trait ParamGuard {
    /// The checked hyperparameters
    type Checked;
    /// Error type resulting from failed hyperparameter checking
    type Error: Error;

    /// Checks the hyperparameters and returns a reference to the checked hyperparameters if
    /// successful
    fn check_ref(&self) -> Result<&Self::Checked, Self::Error>;

    /// Checks the hyperparameters and returns the checked hyperparameters if successful
    fn check(self) -> Result<Self::Checked, Self::Error>;

    /// Calls `check()` and unwraps the result
    fn check_unwrap(self) -> Self::Checked
    where
        Self: Sized,
    {
        self.check().unwrap()
    }
}

/// Implement this trait to opt into a blanket `Transformer` impl that wraps the output of the
/// unchecked `transform` call in a `Result`. If the unchecked `transform` call returns a `Result`,
/// the blanket impl will return double `Result`s, so this trait should be avoided in that case.
pub trait TransformGuard: ParamGuard {}

/// Performs the checking step and calls `transform` on the checked hyperparameters. Returns error
/// if checking was unsuccessful.
impl<R: Records, T, P: TransformGuard> Transformer<R, Result<T, P::Error>> for P
where
    P::Checked: Transformer<R, T>,
{
    fn transform(&self, x: R) -> Result<T, P::Error> {
        self.check_ref().map(|p| p.transform(x))
    }
}

/// Performs checking step and calls `fit` on the checked hyperparameters. If checking failed, the
/// checking error is converted to the original error type of `Fit` and returned.
impl<R: Records, T, E, P: ParamGuard> Fit<R, T, E> for P
where
    P::Checked: Fit<R, T, E>,
    E: Error + From<crate::error::Error> + From<P::Error>,
{
    type Object = <<P as ParamGuard>::Checked as Fit<R, T, E>>::Object;

    fn fit(&self, dataset: &crate::DatasetBase<R, T>) -> Result<Self::Object, E> {
        let checked = self.check_ref()?;
        checked.fit(dataset)
    }
}

/// Performs checking step and calls `fit_with` on the checked hyperparameters. If checking failed,
/// the checking error is converted to the original error type of `FitWith` and returned.
impl<'a, R: Records, T, E, P: ParamGuard> FitWith<'a, R, T, E> for P
where
    P::Checked: FitWith<'a, R, T, E>,
    E: Error + From<crate::error::Error> + From<P::Error>,
{
    type ObjectIn = <<P as ParamGuard>::Checked as FitWith<'a, R, T, E>>::ObjectIn;
    type ObjectOut = <<P as ParamGuard>::Checked as FitWith<'a, R, T, E>>::ObjectOut;

    fn fit_with(
        &self,
        model: Self::ObjectIn,
        dataset: &'a crate::DatasetBase<R, T>,
    ) -> Result<Self::ObjectOut, E> {
        let checked = self.check_ref()?;
        checked.fit_with(model, dataset)
    }
}
