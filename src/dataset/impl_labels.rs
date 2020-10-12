use super::{Targets, Label, Dataset, Data};
use ndarray::{Dimension, ArrayBase};
use std::collections::HashSet;

impl<L: Label> Targets for Vec<L> {
    type Elem = L;

    fn labels(&self) -> HashSet<&L> {
        self.iter().collect()
    }
}

impl<L: Label> Targets for &[L] {
    type Elem = L;

    fn labels(&self) -> HashSet<&L> {
        self.iter().collect()
    }
}

impl<L: Label, S: ndarray::Data<Elem = L>, I: Dimension> Targets for ArrayBase<S, I> {
    type Elem = L;

    fn labels(&self) -> HashSet<&L> {
        self.iter().collect()
    }
}

pub struct TargetsWithLabels<L: Label, T: Targets<Elem = L>> {
    targets: T,
    labels: HashSet<L>
}

impl<L: Label, T: Targets<Elem = L>> Targets for TargetsWithLabels<L, T> {
    type Elem = L;

    fn labels(&self) -> HashSet<&L> {
        self.labels.iter().collect()
    }
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
