use std::borrow::Cow;
use std::error::Error;

use crate::{
    prelude::Records,
    traits::{Fit, IncrementalFit, Transformer},
};

/// A set of parameters whose values have not been checked for validity. A reference to the
/// checked hyperparameters can only be obtained after checking has completed. If the
/// `Transformer`, `Fit`, or `IncrementalFit` traits have been implemented on the checked
/// hyperparameters, they will also be implemented on the unchecked hyperparameters with the
/// checking step done automatically.
///
/// The hyperparameter validation done in `check_ref()` and `check()` should be identical.
pub trait ParamGuard: ParamIntoChecked {
    type Error: Error;

    fn is_valid(&self) -> Result<(), Self::Error>;

    /// Check the parameter set and returns an error if an invalid value is encountered
    fn check_ref<'a>(&'a self) -> Result<Cow<'a, Self::Checked>, Self::Error> {
        self.is_valid().map(|_| self.as_checked())
    }

    /// Checks the hyperparameters and returns the checked hyperparameters if successful
    fn check(self) -> Result<Self::Checked, Self::Error>
    where
        Self: Sized,
    {
        self.is_valid().map(|_| self.into_checked())
    }

    /// Calls `check()` and unwraps the result
    fn check_unwrap(self) -> Self::Checked
    where
        Self: Sized,
    {
        self.check().unwrap()
    }
}

pub trait ParamIntoChecked {
    type Checked: Clone;

    fn as_checked<'a>(&'a self) -> Cow<'a, Self::Checked>;
    fn into_checked(self) -> Self::Checked;
}

pub unsafe trait ParamIntoCheckedConst {
    type Checked: Clone;
}

impl<P> ParamIntoChecked for P
where
    P: ParamIntoCheckedConst,
{
    type Checked = P::Checked;

    fn as_checked<'a>(&'a self) -> Cow<'a, Self::Checked> {
        Cow::Borrowed(unsafe { &*(self as *const Self as *const Self::Checked) })
    }

    fn into_checked(self) -> Self::Checked {
        unsafe { std::ptr::read(&self as *const Self as *const Self::Checked) }
    }
}

/// Performs the checking step and calls `transform` on the checked hyperparameters. Returns error
/// if checking was unsuccessful.
impl<R: Records, T, P: ParamGuard> Transformer<R, Result<T, P::Error>> for P
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
    type Object = <<P as ParamIntoChecked>::Checked as Fit<R, T, E>>::Object;

    fn fit(&self, dataset: &crate::DatasetBase<R, T>) -> Result<Self::Object, E> {
        let checked = self.check_ref()?;
        checked.fit(dataset)
    }
}

/// Performs checking step and calls `fit_with` on the checked hyperparameters. If checking failed,
/// the checking error is converted to the original error type of `IncrementalFit` and returned.
impl<'a, R: Records, T, E, P: ParamGuard> IncrementalFit<'a, R, T, E> for P
where
    P::Checked: IncrementalFit<'a, R, T, E>,
    E: Error + From<crate::error::Error> + From<P::Error>,
{
    type ObjectIn = <<P as ParamIntoChecked>::Checked as IncrementalFit<'a, R, T, E>>::ObjectIn;
    type ObjectOut = <<P as ParamIntoChecked>::Checked as IncrementalFit<'a, R, T, E>>::ObjectOut;

    fn fit_with(
        &self,
        model: Self::ObjectIn,
        dataset: &'a crate::DatasetBase<R, T>,
    ) -> Result<Self::ObjectOut, E> {
        let checked = self.check_ref()?;
        checked.fit_with(model, dataset)
    }
}
