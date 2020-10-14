use ndarray::Array2;
use std::collections::HashSet;
use super::{Float, Label, Dataset, iter::Iter, Records, Targets};

impl<F: Float, L: Label> Dataset<Array2<F>, Vec<L>> {
    pub fn iter<'a>(&'a self) -> Iter<'a, Array2<F>, Vec<L>> {
        Iter::new(&self.records, &self.targets)
    }
}

impl<R: Records, S: Targets> Dataset<R, S> {
    pub fn labels(&self) -> HashSet<&S::Elem> {
        self.targets.labels()
    }

    pub fn targets(&self) -> &[S::Elem] {
        self.targets.as_slice()
    }

    pub fn with_records<T: Records>(self, records: T) -> Dataset<T, S> {
        Dataset {
            records,
            targets: self.targets
        }
    }
}

