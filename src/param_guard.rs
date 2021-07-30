use std::borrow::Cow;
use std::error::Error;

use crate::{
    prelude::Records,
    traits::{Fit, FitWith, Transformer},
};

/// A set of parameters whose values have not been checked for validity. A reference to the
/// checked hyperparameters can only be obtained after checking has completed. If the
/// `Transformer`, `Fit`, or `FitWith` traits have been implemented on the checked
/// hyperparameters, they will also be implemented on the unchecked hyperparameters with the
/// checking step done automatically.
///
/// The hyperparameter validation done in `check_ref()` and `check()` should be identical.
pub trait Verify: ParamIntoChecked {
    type Error: Error;
    type Parameter;

    fn is_valid(&self) -> Result<(), Self::Error>;

    /// Check the parameter set and returns an error if an invalid value is encountered
    fn check_ref<'a>(&'a self) -> Result<&'a Self::Parameter, Self::Error>;

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
    type Checked;

    fn into_checked(self) -> Self::Checked;
}

pub struct Guarded<T>(pub(crate) T);

impl<T> ParamIntoChecked for Guarded<T> {
    type Checked = T;

    fn into_checked(self) -> T {
        self.0
    }
}

impl<T: Verify> Verify for Guarded<T> {
    type Error = T::Error;
    type Parameter = T;

    fn is_valid(&self) -> Result<(), T::Error> {
        Ok(())
    }

    fn check_ref(&self) -> Result<&T, T::Error> {
        Ok(&self.0)
    }
}

/*
/// Performs the checking step and calls `transform` on the checked hyperparameters. Returns error
/// if checking was unsuccessful.
impl<R: Records, T, P: Verify> Transformer<R, Result<T, P::Error>> for P
where
    P::Checked: Transformer<R, T>,
{
    fn transform(&self, x: R) -> Result<T, P::Error> {
        self.check_ref().map(|p| p.transform(x))
    }
}

/// Performs checking step and calls `fit` on the checked hyperparameters. If checking failed, the
/// checking error is converted to the original error type of `Fit` and returned.
impl<R: Records, T, E, P: Verify> Fit<R, T, E> for P
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
/// the checking error is converted to the original error type of `FitWith` and returned.
impl<'a, R: Records, T, E, P: Verify> FitWith<'a, R, T, E> for P
where
    P::Checked: FitWith<'a, R, T, E>,
    E: Error + From<crate::error::Error> + From<P::Error>,
{
    type ObjectIn = <<P as ParamIntoChecked>::Checked as FitWith<'a, R, T, E>>::ObjectIn;
    type ObjectOut = <<P as ParamIntoChecked>::Checked as FitWith<'a, R, T, E>>::ObjectOut;

    fn fit_with(
        &self,
        model: Self::ObjectIn,
        dataset: &'a crate::DatasetBase<R, T>,
    ) -> Result<Self::ObjectOut, E> {
        let checked = self.check_ref()?;
        checked.fit_with(model, dataset)
    }
}*/
