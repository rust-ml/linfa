use ndarray::Array2;
use std::collections::HashSet;
use super::{Float, Label, Dataset, iter::Iter, Records, Targets, Labels};

impl<F: Float, L: Label> Dataset<Array2<F>, Vec<L>> {
    pub fn iter<'a>(&'a self) -> Iter<'a, Array2<F>, Vec<L>> {
        Iter::new(&self.records, &self.targets)
    }
}

impl<R: Records, S: Targets> Dataset<R, S> {
    pub fn targets(&self) -> &[S::Elem] {
        self.targets.as_slice()
    }

    pub fn with_records<T: Records>(self, records: T) -> Dataset<T, S> {
        Dataset {
            records,
            targets: self.targets
        }
    }

    pub fn map_targets<T, G: FnMut(&S::Elem) -> T>(self, fnc: G) -> Dataset<R, Vec<T>> {
        let Dataset { records, targets } = self;

        let new_targets = targets.as_slice().into_iter()
            .map(fnc)
            .collect::<Vec<T>>();

        Dataset {
            records,
            targets: new_targets
        }
    }
}

impl<R: Records, S: Labels> Dataset<R, S> {
    pub fn labels(&self) -> HashSet<&S::Elem> {
        self.targets.labels()
    }
}

impl<F: Float> From<Array2<F>> for Dataset<Array2<F>, ()> {
    fn from(records: Array2<F>) -> Self {
        Dataset {
            records,
            targets: ()
        }
    }
}

impl<F: Float, T: Targets> From<(Array2<F>, T)> for Dataset<Array2<F>, T> {
    fn from(rec_tar: (Array2<F>, T)) -> Self {
        Dataset {
            records: rec_tar.0,
            targets: rec_tar.1
        }
    }
}
