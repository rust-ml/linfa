use ndarray::Array2;
use super::{Float, Label, Dataset, iter::Iter, Records, Targets, Labels};

impl<F: Float, L: Label> Dataset<Array2<F>, Vec<L>> {
    pub fn iter<'a>(&'a self) -> Iter<'a, Array2<F>, Vec<L>> {
        Iter::new(&self.records, &self.targets)
    }
}

impl<R: Records, S: Targets> Dataset<R, S> {
    pub fn new(records: R, targets: S) -> Dataset<R, S> {
        Dataset {
            records,
            targets,
            labels: Vec::new(),
            weights: Vec::new()
        }
    }

    pub fn targets(&self) -> &[S::Elem] {
        self.targets.as_slice()
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn with_records<T: Records>(self, records: T) -> Dataset<T, S> {
        Dataset {
            records,
            targets: self.targets,
            labels: self.labels,
            weights: Vec::new()
        }
    }

    pub fn map_targets<T, G: FnMut(&S::Elem) -> T>(self, fnc: G) -> Dataset<R, Vec<T>> {
        let Dataset { records, targets, labels, weights } = self;

        let new_targets = targets.as_slice().into_iter()
            .map(fnc)
            .collect::<Vec<T>>();

        Dataset {
            records,
            targets: new_targets,
            labels: Vec::new(),
            weights
        }
    }
}

impl<R: Records, S: Targets + Labels> Dataset<R, S> {
    pub fn labels(&self) -> Vec<<S as Labels>::Elem> {
        self.targets.labels()
    }
}

impl<F: Float> From<Array2<F>> for Dataset<Array2<F>, ()> {
    fn from(records: Array2<F>) -> Self {
        Dataset {
            records,
            targets: (),
            labels: Vec::new(),
            weights: Vec::new()
        }
    }
}

impl<F: Float, T: Targets> From<(Array2<F>, T)> for Dataset<Array2<F>, T> {
    fn from(rec_tar: (Array2<F>, T)) -> Self {
        Dataset {
            records: rec_tar.0,
            targets: rec_tar.1,
            labels: Vec::new(),
            weights: Vec::new()
        }
    }
}
