use super::{iter::Iter, Dataset, Float, Label, Labels, Records, Targets};
use ndarray::{Axis, Array2, ArrayView2, ArrayBase, Data, Dimension};

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
            weights: Vec::new(),
        }
    }

    pub fn with_targets<T: Targets>(self, targets: T) -> Dataset<R, T> {
        Dataset {
            records: self.records,
            targets,
            weights: self.weights
        }
    }

    pub fn with_weights(mut self, weights: Vec<f32>) -> Dataset<R, S> {
        self.weights = weights;

        self
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

impl<R: Records, S: Labels> Dataset<R, S> {
    pub fn labels(&self) -> Vec<S::Elem> {
        self.targets.labels()
    }
}

impl<F: Float, D: Data<Elem = F>, I: Dimension> From<ArrayBase<D, I>> for Dataset<ArrayBase<D, I>, ()> {
    fn from(records: ArrayBase<D, I>) -> Self {
        Dataset {
            records,
            targets: (),
            weights: Vec::new(),
        }
    }
}

impl<F: Float, T: Targets, D: Data<Elem = F>, I: Dimension> From<(ArrayBase<D, I>, T)> for Dataset<ArrayBase<D, I>, T> {
    fn from(rec_tar: (ArrayBase<D, I>, T)) -> Self {
        Dataset {
            records: rec_tar.0,
            targets: rec_tar.1,
            weights: Vec::new(),
        }
    }
}
