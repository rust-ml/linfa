//! Datasets
//!
//! This module implements the dataset struct and various helper traits to extend its
//! functionality.
use ndarray::{NdFloat, ArrayBase, Ix2, Ix1, ArrayView};
use num_traits::{FromPrimitive, Signed};
use std::cmp::{Ordering, PartialOrd};
use std::hash::Hash;
use std::iter::Sum;
use std::ops::Deref;

mod impl_dataset;
mod impl_records;
mod impl_targets;

mod iter;

/// Floating numbers
pub trait Float: NdFloat + FromPrimitive + Signed + Default + Sum {}
impl Float for f32 {}
impl Float for f64 {}

/// Discrete labels
pub trait Label: PartialEq + Eq + Hash + Clone {}

impl Label for bool {}
impl Label for usize {}
impl Label for String {}

/// Probability types
///
/// This helper struct exists to distinguish probabilities from floating points. For example SVM
/// selects regression or classification training, based on the target type, and could not
/// distinguish them with floating points alone.
#[derive(Debug, Copy, Clone)]
pub struct Pr(pub f32);

impl PartialEq for Pr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for Pr {
    fn partial_cmp(&self, other: &Pr) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Deref for Pr {
    type Target = f32;

    fn deref(&self) -> &f32 {
        &self.0
    }
}

/// DatasetBase
///
/// A dataset contains a number of records and targets. Each record corresponds to a single target
/// and may be weighted with the `weights` field during the training process.
pub struct DatasetBase<R, T>
where
    R: Records,
    T: Targets,
{
    pub records: R,
    pub targets: T,

    weights: Vec<f32>,
}

/// Dataset
/// 
/// The most commonly used typed of dataset. It contains a number of records 
/// stored as an `Array2` and each record corresponds to a single target. Such 
/// targets are stored as an `Array1`.
pub type Dataset<D> = DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>; 

/// DatasetView
/// 
/// A read only view of a Dataset
pub type DatasetView<'a, D> = DatasetBase<ArrayView<'a, D, Ix2>, ArrayView<'a, D, Ix2>>;


/// Records
///
/// The records are input data in the training
pub trait Records: Sized {
    type Elem;

    fn observations(&self) -> usize;
}

/// Targets
pub trait Targets {
    type Elem;

    fn as_slice(&self) -> &[Self::Elem];
}

/// Labels
///
/// Same as targets, but with discrete elements. The labels trait can therefore return the set of
/// labels of the targets
pub trait Labels: Targets
where
    Self::Elem: Label,
{
    fn labels(&self) -> Vec<Self::Elem>;
}
