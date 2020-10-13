use std::iter::Sum;
use std::hash::Hash;
use std::collections::HashSet;
use ndarray::NdFloat;
use num_traits::FromPrimitive;

mod impl_dataset;
mod impl_targets;
mod impl_records;

mod iter;

pub trait Float: NdFloat + FromPrimitive + Default + Sum {}
impl Float for f32 {}
impl Float for f64 {}

pub trait Label: PartialEq + Eq + Hash {}

impl Label for bool {}
impl Label for usize {}
impl Label for String {}

/// Probability types
pub type Pr = f32;

pub struct Dataset<R, S>
where
    R: Records
{
    pub records: R,
    pub targets: S,
}

pub trait Records: Sized {
    type Elem;

    fn observations(&self) -> usize;
}

pub trait Targets {
    type Elem;

    fn labels<'a>(&'a self) -> HashSet<&'a Self::Elem>;
    fn as_slice<'a>(&'a self) -> &'a [Self::Elem];
}
