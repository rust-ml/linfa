use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Dimension, Ix2};
use rand::{seq::SliceRandom, Rng};
use std::collections::HashMap;

use super::{iter::Iter, Dataset, Float, Label, Labels, Records, Targets};

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

    pub fn target(&self, idx: usize) -> &S::Elem {
        &self.targets.as_slice()[idx]
    }

    pub fn weights(&self) -> Option<&[f32]> {
        if self.weights.len() > 0 {
            Some(&self.weights)
        } else {
            None
        }
    }

    pub fn weight_for(&self, idx: usize) -> f32 {
        self.weights.get(idx).map(|x| *x).unwrap_or(1.0)
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

impl<F: Float, T: Clone> Dataset<Array2<F>, Vec<T>> {
    pub fn shuffle<R: Rng>(self, mut rng: &mut R) -> Self {
        let mut indices = (0..self.observations()).collect::<Vec<_>>();
        indices.shuffle(&mut rng);

        let records = self.records().select(Axis(0), &indices);
        let targets = indices
            .iter()
            .map(|x| self.targets[*x].clone())
            .collect::<Vec<_>>();

        Dataset::new(records, targets)
    }

    pub fn bootstrap<'a, R: Rng>(
        &'a self,
        num_samples: usize,
        rng: &'a mut R,
    ) -> impl Iterator<Item = Dataset<Array2<F>, Vec<T>>> + 'a {
        std::iter::repeat(()).map(move |_| {
            // sample with replacement
            let indices = (0..num_samples)
                .map(|_| rng.gen_range(0, self.observations()))
                .collect::<Vec<_>>();

            let records = self.records().select(Axis(0), &indices);
            let targets = indices
                .iter()
                .map(|x| self.targets.as_slice()[*x].clone())
                .collect::<Vec<_>>();

            Dataset::new(records, targets)
        })
    }
}

impl<F: Float, T: Clone> Dataset<Array2<F>, Array1<T>> {
    pub fn shuffle<R: Rng>(self, mut rng: &mut R) -> Self {
        let mut indices = (0..self.observations()).collect::<Vec<_>>();
        indices.shuffle(&mut rng);

        let records = self.records().select(Axis(0), &indices);
        let targets = indices
            .iter()
            .map(|x| self.targets[*x].clone())
            .collect::<Array1<_>>();

        Dataset::new(records, targets)
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

impl<L: Label, R: Records, S: Labels<Elem = L>> Dataset<R, S> {
    pub fn labels(&self) -> Vec<L> {
        self.targets.labels()
    }

    pub fn frequencies_with_mask(&self, mask: &[bool]) -> HashMap<&L, f32> {
        let mut freqs = HashMap::new();

        for (elm, val) in self
            .targets
            .as_slice()
            .iter()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(i, x)| (x, self.weight_for(i)))
        {
            *freqs.entry(elm).or_insert(0.0) += val;
        }

        freqs
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
