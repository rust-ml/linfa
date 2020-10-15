use super::{Targets, Label, Dataset, Records, Labels};
use ndarray::{Dimension, ArrayBase, Data};
impl<L> Targets for Vec<L> {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        &self
    }
}

impl<L: Label> Labels for Vec<L> {
    type Elem = L;

    fn labels(&self) -> Vec<L> {
        self.iter().cloned().collect()
    }
}

impl<L> Targets for &Vec<L> {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<L: Label> Labels for &Vec<L> {
    type Elem = L;

    fn labels(&self) -> Vec<L> {
        self.iter().cloned().collect()
    }
}


impl<L> Targets for &[L] {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}


impl<L: Label> Labels for &[L] {
    type Elem = L;

    fn labels(&self) -> Vec<L> {
        self.iter().cloned().collect()
    }
}

impl<L, S: Data<Elem = L>, I: Dimension> Targets for ArrayBase<S, I> {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        self.as_slice().unwrap()
    }
}

impl<L: Label, S: Data<Elem = L>, I: Dimension> Labels for ArrayBase<S, I> {
    type Elem = L;

    fn labels(&self) -> Vec<L> {
        self.iter().cloned().collect()
    }
}

impl Targets for () {
    type Elem = ();

    fn as_slice(&self) -> &[()] {
        &[()]
    }
}

impl<R: Records, L: Label, T: Targets<Elem=L>> Dataset<R, T> {
    pub fn with_labels(self, labels: Vec<L>) -> Dataset<R, T> {
        Dataset {
            records: self.records,
            targets: self.targets,
            weights: self.weights,
            labels
        }
    }
}
