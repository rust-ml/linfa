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

pub trait Label: PartialEq + Eq + Hash + Clone {}

impl Label for bool {}
impl Label for usize {}
impl Label for String {}

/// Probability types
pub type Pr = f32;

pub struct Dataset<R, T>
where
    R: Records,
    T: Targets
{
    pub records: R,
    pub targets: T,

    labels: Vec<T::Elem>,
    weights: Vec<f32>
}

pub trait Records: Sized {
    type Elem;

    fn observations(&self) -> usize;
}

pub trait Targets {
    type Elem;

    fn as_slice<'a>(&'a self) -> &'a [Self::Elem];
}

pub trait Labels {
    type Elem: Label;

    fn labels(&self) -> Vec<Self::Elem>;
}
