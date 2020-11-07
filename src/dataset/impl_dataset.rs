use super::{iter::Iter, Dataset, Float, Label, Labels, Records, Targets};
use ndarray::{Array2, ArrayBase, ArrayView2, Axis, Data, Dimension, Ix2};

impl<F: Float, L: Label> Dataset<Array2<F>, Vec<L>> {
    pub fn iter(&self) -> Iter<'_, Array2<F>, Vec<L>> {
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
            weights: self.weights,
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

        let new_targets = targets.as_slice().iter().map(fnc).collect::<Vec<T>>();

        Dataset {
            records,
            targets: new_targets,
            weights,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<F: Float, T: Targets, D: Data<Elem = F>> Dataset<ArrayBase<D, Ix2>, T> {
    pub fn split_with_ratio(
        &self,
        ratio: f32,
    ) -> (
        Dataset<ArrayView2<'_, F>, &[T::Elem]>,
        Dataset<ArrayView2<'_, F>, &[T::Elem]>,
    ) {
        let n = (self.observations() as f32 * ratio).ceil() as usize;
        let (first, second) = self.records.view().split_at(Axis(0), n);

        let targets = self.targets();
        let (first_targets, second_targets) = (&targets[..n], &targets[n..]);

        let dataset1 = Dataset::new(first, first_targets);
        let dataset2 = Dataset::new(second, second_targets);

        (dataset1, dataset2)
    }
}

impl<F: Float, L: Label, T: Labels<Elem = L>, D: Data<Elem = F>> Dataset<ArrayBase<D, Ix2>, T> {
    pub fn one_vs_all(&self) -> Vec<Dataset<ArrayView2<'_, F>, Vec<bool>>> {
        self.labels()
            .into_iter()
            .map(|label| {
                let targets = self
                    .targets()
                    .as_slice()
                    .iter()
                    .map(|x| x == &label)
                    .collect();

                Dataset::new(self.records().view(), targets)
            })
            .collect()
    }
}

impl<R: Records, S: Labels> Dataset<R, S> {
    pub fn labels(&self) -> Vec<S::Elem> {
        self.targets.labels()
    }
}

impl<F: Float, D: Data<Elem = F>, I: Dimension> From<ArrayBase<D, I>>
    for Dataset<ArrayBase<D, I>, ()>
{
    fn from(records: ArrayBase<D, I>) -> Self {
        Dataset {
            records,
            targets: (),
            weights: Vec::new(),
        }
    }
}

impl<F: Float, T: Targets, D: Data<Elem = F>, I: Dimension> From<(ArrayBase<D, I>, T)>
    for Dataset<ArrayBase<D, I>, T>
{
    fn from(rec_tar: (ArrayBase<D, I>, T)) -> Self {
        Dataset {
            records: rec_tar.0,
            targets: rec_tar.1,
            weights: Vec::new(),
        }
    }
}
