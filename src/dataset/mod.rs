use std::hash::Hash;
use ndarray::NdFloat;
use num_traits::FromPrimitive;

mod impl_dataset;
mod impl_labels;
mod impl_data;
mod iter;

pub trait Float: NdFloat + FromPrimitive + Default {}
impl Float for f32 {}
impl Float for f64 {}

pub trait Label: PartialEq + Eq + Hash {}

impl Label for bool {}
impl Label for usize {}
impl Label for String {}

pub struct Dataset<T, S>
where
    T: Data,
{
    data: T,
    targets: S,
}

pub trait Data: Sized {
    type Elem;

    fn observations(&self) -> usize;
}

pub trait Targets {
    type Elem;
}
