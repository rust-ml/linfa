use super::{Targets, Label, Dataset, Data};
use ndarray::{Dimension, ArrayBase};
use std::collections::HashSet;

impl<L: Label> Targets for Vec<L> {
    type Elem = L;
}

impl<L: Label> Targets for &[L] {
    type Elem = L;
}

impl<L: Label, S: ndarray::Data<Elem = L>, I: Dimension> Targets for ArrayBase<S, I> {
    type Elem = L;
}

pub struct TargetsWithLabels<L: Label, T: Targets<Elem = L>> {
    targets: T,
    labels: HashSet<L>
}

impl<L: Label, T: Targets<Elem = L>> Targets for TargetsWithLabels<L, T> {
    type Elem = L;
}

impl<D: Data, L: Label, T: Targets<Elem=L>> Dataset<D, T> {
    pub fn with_labels(self, labels: Vec<L>) -> Dataset<D, TargetsWithLabels<L, T>> {
        let targets = TargetsWithLabels {
            targets: self.targets,
            labels: labels.into_iter().collect()
        };

        Dataset {
            data: self.data,
            targets
        }
    }
}

impl<D: Data, L: Label> Dataset<D, Vec<L>> {
    pub fn labels(&self) -> HashSet<&L> {
        self.targets.iter().collect()
    }
}

impl<D: Data, L: Label, T: Targets<Elem = L>> Dataset<D, TargetsWithLabels<L, T>> {
    pub fn labels(&self) -> &HashSet<L> {
        &self.targets.labels
    }
}
