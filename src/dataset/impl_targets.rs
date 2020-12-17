use super::{DatasetBase, Label, Labels, Records, Targets};
use ndarray::{ArrayBase, Data, Ix1};
use std::collections::HashSet;

/// A vector can act as targets
impl<L> Targets for Vec<L> {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        &self
    }
}

/// A vector with discrete labels can act as labels
impl<L: Label> Labels for Vec<L> {
    fn labels(&self) -> Vec<L> {
        self.iter()
            .cloned()
            .collect::<HashSet<L>>()
            .into_iter()
            .collect()
    }
}

/// A slice can act as targets
impl<L> Targets for &[L] {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

/// A slice with discrete labels can act as labels
impl<L: Label> Labels for &[L] {
    fn labels(&self) -> Vec<L> {
        self.iter()
            .cloned()
            .collect::<HashSet<L>>()
            .into_iter()
            .collect()
    }
}

/// A NdArray can act as targets
impl<L, S: Data<Elem = L>> Targets for ArrayBase<S, Ix1> {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        self.as_slice().unwrap()
    }
}

/// A NdArray with discrete labels can act as labels
impl<L: Label, S: Data<Elem = L>> Labels for ArrayBase<S, Ix1> {
    fn labels(&self) -> Vec<L> {
        self.iter()
            .cloned()
            .collect::<HashSet<L>>()
            .into_iter()
            .collect()
    }
}

/// Empty targets for datasets with just observations
impl Targets for () {
    type Elem = ();

    fn as_slice(&self) -> &[()] {
        &[()]
    }
}

impl<T: Targets> Targets for &T {
    type Elem = T::Elem;

    fn as_slice(&self) -> &[Self::Elem] {
        (*self).as_slice()
    }
}

impl<L: Label, T: Labels<Elem = L>> Labels for &T {
    fn labels(&self) -> Vec<T::Elem> {
        (*self).labels()
    }
}

/// Targets with precomputed labels
pub struct TargetsWithLabels<P: Label, L: Labels<Elem = P>> {
    targets: L,
    labels: HashSet<L::Elem>,
}

impl<L: Label, T: Labels<Elem = L>> Targets for TargetsWithLabels<L, T> {
    type Elem = T::Elem;

    fn as_slice(&self) -> &[Self::Elem] {
        self.targets.as_slice()
    }
}

impl<L: Label + Clone, T: Labels<Elem = L>> Labels for TargetsWithLabels<L, T> {
    fn labels(&self) -> Vec<T::Elem> {
        self.labels.iter().cloned().collect()
    }
}

impl<R: Records, L: Label, T: Labels<Elem = L>> DatasetBase<R, T> {
    pub fn with_labels(self, labels: &[L]) -> DatasetBase<R, TargetsWithLabels<L, T>> {
        let targets = TargetsWithLabels {
            targets: self.targets,
            labels: labels.iter().cloned().collect(),
        };

        DatasetBase {
            records: self.records,
            weights: self.weights,
            targets,
        }
    }
}
