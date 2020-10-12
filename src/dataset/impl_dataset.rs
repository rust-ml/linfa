use ndarray::Array2;
use std::collections::HashSet;
use super::{Float, Label, Dataset, iter::Iter, Data, Targets};

impl<F: Float, L: Label> Dataset<Array2<F>, Vec<L>> {
    pub fn iter<'a>(&'a self) -> Iter<'a, Array2<F>, Vec<L>> {
        Iter::new(&self.data, &self.targets)
    }
}

impl<T: Data, S: Targets> Dataset<T, S> {
    pub fn labels(&self) -> HashSet<&S::Elem> {
        self.targets.labels()
    }
}

