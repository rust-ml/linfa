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

    fn verify(&self) -> Result<(), Self::Error>;

    /// Checks the hyperparameters and returns the checked hyperparameters if successful
    fn check(self) -> Result<Self::Checked, Self::Error>
    where
        Self: Sized,
    {
        self.verify().map(|_| self.into_checked())
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
