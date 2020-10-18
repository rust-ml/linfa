use super::{iter::Iter, Dataset, Float, Label, Labels, Records, Targets};
use ndarray::{Axis, Array2, ArrayView2, ArrayBase, Ix2, Data};

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
            weights: Vec::new(),
        }
    }

    pub fn targets(&self) -> &[S::Elem] {
        self.targets.as_slice()
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn records(&self) -> &R {
        &self.records
    }

    pub fn with_records<T: Records>(self, records: T) -> Dataset<T, S> {
        Dataset {
            records,
            targets: self.targets,
            labels: self.labels,
            weights: Vec::new(),
        }
    }

    pub fn with_targets<T: Targets>(self, targets: T) -> Dataset<R, T> {
        Dataset {
            records: self.records,
            targets,
            labels: Vec::new(),
            weights: self.weights
        }
    }

    pub fn map_targets<T, G: FnMut(&S::Elem) -> T>(self, fnc: G) -> Dataset<R, Vec<T>> {
        let Dataset {
            records,
            targets,
            weights,
            ..
        } = self;

        let new_targets = targets.as_slice().into_iter().map(fnc).collect::<Vec<T>>();

        Dataset {
            records,
            targets: new_targets,
            labels: Vec::new(),
            weights,
        }
    }
}

impl<F: Float, T: Targets> Dataset<Array2<F>, T> {
    pub fn split_with_ratio<'a>(&'a self, ratio: f32) -> (Dataset<ArrayView2<'a, F>, &'a [T::Elem]>, Dataset<ArrayView2<'a, F>, &'a [T::Elem]>) {
        let n = (self.observations() as f32 * ratio).ceil() as usize;
        let (a, b) = self.records.view().split_at(Axis(0), n);

        let targets = self.targets();
        let (c, d) = (&targets[..n], &targets[n..]);

        let d1 = Dataset::new(a, c);
        let d2 = Dataset::new(b, d);

        (d1, d2)
    }
}

impl<'a, F: Float, T: Targets> Dataset<ArrayView2<'a, F>, T> {
    pub fn split_with_ratio(&'a self, ratio: f32) -> (Dataset<ArrayView2<'a, F>, &'a [T::Elem]>, Dataset<ArrayView2<'a, F>, &'a [T::Elem]>) {
        let n = (self.observations() as f32 * ratio).ceil() as usize;
        let (a, b) = self.records.split_at(Axis(0), n);

        let targets = self.targets();
        let (c, d) = (&targets[..n], &targets[n..]);

        let d1 = Dataset::new(a, c);
        let d2 = Dataset::new(b, d);

        (d1, d2)
    }
}

impl<R: Records, S: Targets + Labels> Dataset<R, S> {
    pub fn labels(&self) -> Vec<<S as Labels>::Elem> {
        self.targets.labels()
    }
}

impl<F: Float, D: Data<Elem = F>> From<ArrayBase<D, Ix2>> for Dataset<ArrayBase<D, Ix2>, ()> {
    fn from(records: ArrayBase<D, Ix2>) -> Self {
        Dataset {
            records,
            targets: (),
            labels: Vec::new(),
            weights: Vec::new(),
        }
    }
}

impl<F: Float, T: Targets> From<(Array2<F>, T)> for Dataset<Array2<F>, T> {
    fn from(rec_tar: (Array2<F>, T)) -> Self {
        Dataset {
            records: rec_tar.0,
            targets: rec_tar.1,
            labels: Vec::new(),
            weights: Vec::new(),
        }
    }
}
