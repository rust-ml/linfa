use super::{Dataset, Label, Labels, Records, Targets};
use std::collections::HashSet;
use ndarray::{ArrayBase, Data, Ix1};

impl<L> Targets for Vec<L> {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        &self
    }
}

impl<L: Label> Labels for Vec<L> {
    fn labels(&self) -> Vec<L> {
        self.iter().cloned().collect::<HashSet<L>>()
            .into_iter().collect()
    }
}

/*impl<L> Targets for &Vec<L> {
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
}*/

impl<L> Targets for &[L] {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<L: Label> Labels for &[L] {
    fn labels(&self) -> Vec<L> {
        self.iter().cloned().collect::<HashSet<L>>()
            .into_iter().collect()
    }
}

impl<L, S: Data<Elem = L>> Targets for ArrayBase<S, Ix1> {
    type Elem = L;

    fn as_slice(&self) -> &[Self::Elem] {
        self.as_slice().unwrap()
    }
}

impl<L: Label, S: Data<Elem = L>> Labels for ArrayBase<S, Ix1> {
    fn labels(&self) -> Vec<L> {
        self.iter().cloned().collect::<HashSet<L>>()
            .into_iter().collect()
    }
}

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

impl<T: Labels> Labels for &T {
    fn labels(&self) -> Vec<T::Elem> {
        (*self).labels()
    }
}

impl<R: Records, L: Label, T: Targets<Elem = L>> Dataset<R, T> {
    pub fn with_labels(self, labels: Vec<L>) -> Dataset<R, T> {
        Dataset {
            records: self.records,
            targets: self.targets,
            weights: self.weights,
            labels,
        }
    }
}
