use super::{Targets, Label, Dataset, Records};
use ndarray::{Dimension, ArrayBase, Data};
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

impl<L: Label, S: Data<Elem = L>, I: Dimension> Targets for ArrayBase<S, I> {
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

impl<R: Records, L: Label, T: Targets<Elem=L>> Dataset<R, T> {
    pub fn with_labels(self, labels: Vec<L>) -> Dataset<R, TargetsWithLabels<L, T>> {
        let targets = TargetsWithLabels {
            targets: self.targets,
            labels: labels.into_iter().collect()
        };

        Dataset {
            records: self.records,
            targets
        }
    }
}
