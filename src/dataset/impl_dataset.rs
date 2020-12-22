use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Dimension, Ix1, Ix2};
use rand::{seq::SliceRandom, Rng};
use std::collections::HashMap;

use super::{
    iter::Iter, Dataset, DatasetBase, DatasetView, Float, Label, Labels, Records, Targets,
};

impl<F: Float, L: Label> DatasetBase<Array2<F>, Vec<L>> {
    pub fn iter(&self) -> Iter<'_, Array2<F>, Vec<L>> {
        Iter::new(&self.records, &self.targets)
    }
}

impl<R: Records, S: Targets> DatasetBase<R, S> {
    pub fn new(records: R, targets: S) -> DatasetBase<R, S> {
        DatasetBase {
            records,
            targets,
            weights: Vec::new(),
        }
    }

    pub fn targets(&self) -> &S {
        &self.targets
    }

    pub fn target(&self, idx: usize) -> &S::Elem {
        &self.targets.as_slice()[idx]
    }

    pub fn weights(&self) -> Option<&[f32]> {
        if !self.weights.is_empty() {
            Some(&self.weights)
        } else {
            None
        }
    }

    pub fn weight_for(&self, idx: usize) -> f32 {
        self.weights.get(idx).copied().unwrap_or(1.0)
    }

    pub fn records(&self) -> &R {
        &self.records
    }

    pub fn with_records<T: Records>(self, records: T) -> DatasetBase<T, S> {
        DatasetBase {
            records,
            targets: self.targets,
            weights: Vec::new(),
        }
    }

    pub fn with_targets<T: Targets>(self, targets: T) -> DatasetBase<R, T> {
        DatasetBase {
            records: self.records,
            targets,
            weights: self.weights,
        }
    }

    pub fn with_weights(mut self, weights: Vec<f32>) -> DatasetBase<R, S> {
        self.weights = weights;

        self
    }
}

impl<F: Float, D: Data<Elem = F>, T: Targets> DatasetBase<ArrayBase<D, Ix2>, T> {
    pub fn map_targets<S, G: FnMut(&T::Elem) -> S>(
        self,
        fnc: G,
    ) -> DatasetBase<ArrayBase<D, Ix2>, Array1<S>> {
        let DatasetBase {
            records,
            targets,
            weights,
            ..
        } = self;

        let new_targets = targets.as_slice().iter().map(fnc).collect::<Vec<S>>();
        let new_targets = Array1::from_shape_vec(new_targets.len(), new_targets).unwrap();

        DatasetBase {
            records,
            targets: new_targets,
            weights,
        }
    }
}

impl<F: Float, T: Clone> DatasetBase<Array2<F>, Vec<T>> {
    pub fn shuffle<R: Rng>(self, mut rng: &mut R) -> Self {
        let mut indices = (0..self.observations()).collect::<Vec<_>>();
        indices.shuffle(&mut rng);

        let records = self.records().select(Axis(0), &indices);
        let targets = indices
            .iter()
            .map(|x| self.targets[*x].clone())
            .collect::<Vec<_>>();

        DatasetBase::new(records, targets)
    }

    pub fn bootstrap<'a, R: Rng>(
        &'a self,
        num_samples: usize,
        rng: &'a mut R,
    ) -> impl Iterator<Item = DatasetBase<Array2<F>, Vec<T>>> + 'a {
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

            DatasetBase::new(records, targets)
        })
    }

    pub fn split_with_ratio(mut self, ratio: f32) -> (Self, Self) {
        let nfeatures = self.records.ncols();
        let npoints = self.records.nrows();
        let n = (npoints as f32 * ratio).ceil() as usize;

        // split records into two disjoint arrays
        let mut array_buf = self.records.into_raw_vec();
        let second_array_buf = array_buf.split_off(n * nfeatures);

        let first = Array2::from_shape_vec((n, nfeatures), array_buf).unwrap();
        let second = Array2::from_shape_vec((npoints - n, nfeatures), second_array_buf).unwrap();

        // split targets into two disjoint Vec
        let second_targets = self.targets.split_off(n);

        // split weights into two disjoint Vec
        let second_weights = if self.weights.len() == npoints {
            self.weights.split_off(n)
        } else {
            vec![]
        };

        // create new datasets with attached weights
        let dataset1 = DatasetBase::new(first, self.targets).with_weights(self.weights);

        let dataset2 = DatasetBase::new(second, second_targets).with_weights(second_weights);

        (dataset1, dataset2)
    }
}

#[allow(clippy::type_complexity)]
impl<F: Float, T: Targets, D: Data<Elem = F>> DatasetBase<ArrayBase<D, Ix2>, T> {
    pub fn split_with_ratio_view(
        &self,
        ratio: f32,
    ) -> (DatasetView<F, T::Elem>, DatasetView<F, T::Elem>) {
        let n = (self.observations() as f32 * ratio).ceil() as usize;
        let (first, second) = self.records.view().split_at(Axis(0), n);
        let targets = self.targets().as_slice();
        let (first_targets, second_targets) = (
            ArrayView1::from(&targets[..n]),
            ArrayView1::from(&targets[n..]),
        );
        let dataset1 = DatasetBase::new(first, first_targets);
        let dataset2 = DatasetBase::new(second, second_targets);
        (dataset1, dataset2)
    }

    pub fn view(&self) -> DatasetView<F, T::Elem> {
        let records = self.records().view();
        let targets = ArrayView1::from(self.targets.as_slice());
        DatasetBase::new(records, targets)
    }
}

impl<F: Float, L: Label, T: Labels<Elem = L>, D: Data<Elem = F>> DatasetBase<ArrayBase<D, Ix2>, T> {
    pub fn one_vs_all(&self) -> Vec<DatasetBase<ArrayView2<'_, F>, Vec<bool>>> {
        self.labels()
            .into_iter()
            .map(|label| {
                let targets = self
                    .targets()
                    .as_slice()
                    .iter()
                    .map(|x| x == &label)
                    .collect();

                DatasetBase::new(self.records().view(), targets)
            })
            .collect()
    }
}

impl<L: Label, R: Records, S: Labels<Elem = L>> DatasetBase<R, S> {
    pub fn labels(&self) -> Vec<L> {
        self.targets.labels()
    }

    /// Calculates label frequencies from a dataset while masking certain samples.
    ///
    /// ### Parameters
    ///
    /// * `mask`: a boolean array that specifies which samples to include in the count
    ///
    /// ### Returns
    ///
    /// A mapping of the Dataset's samples to their frequencies
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
    for DatasetBase<ArrayBase<D, I>, ()>
{
    fn from(records: ArrayBase<D, I>) -> Self {
        DatasetBase {
            records,
            targets: (),
            weights: Vec::new(),
        }
    }
}

impl<F: Float, T: Targets, D: Data<Elem = F>, I: Dimension> From<(ArrayBase<D, I>, T)>
    for DatasetBase<ArrayBase<D, I>, T>
{
    fn from(rec_tar: (ArrayBase<D, I>, T)) -> Self {
        DatasetBase {
            records: rec_tar.0,
            targets: rec_tar.1,
            weights: Vec::new(),
        }
    }
}

impl<F: Float, E: Copy> Dataset<F, E> {
    pub fn bootstrap<'a, R: Rng>(
        &'a self,
        num_samples: usize,
        rng: &'a mut R,
    ) -> impl Iterator<Item = Dataset<F, E>> + 'a {
        std::iter::repeat(()).map(move |_| {
            // sample with replacement
            let indices = (0..num_samples)
                .map(|_| rng.gen_range(0, self.observations()))
                .collect::<Vec<_>>();

            let records = self.records().select(Axis(0), &indices);
            let targets = indices
                .iter()
                .map(|x| self.targets[*x])
                .collect::<ArrayBase<_, Ix1>>();

            Dataset::new(records, targets)
        })
    }

    /// Produces a shuffled version of the current Dataset.
    ///
    /// ### Parameters
    ///
    /// * `rng`: the random number generator that will be used to shuffle the samples
    ///
    /// ### Returns
    ///
    /// A new shuffled version of the current Dataset
    pub fn shuffle<R: Rng>(self, rng: &mut R) -> Self {
        self.view().shuffle(rng)
    }

    /// Splits the current Dataset into two new ones according to the ratio given in input.
    /// If the input Dataset contains `n` samples then the two new Datasets will have respectively
    /// `n * ratio` and `n - (n*ratio)` samples.
    ///
    /// ### Parameters
    ///
    /// * `ratio`: the ratio of samples in the input Dataset to include in the first output one
    ///
    /// ### Returns
    ///  
    /// The input Dataset split into two according to the input ratio.
    pub fn split_with_ratio(mut self, ratio: f32) -> (Self, Self) {
        let nfeatures = self.records.ncols();
        let npoints = self.records.nrows();
        let n = (npoints as f32 * ratio).ceil() as usize;

        // split records into two disjoint arrays
        let mut array_buf = self.records.into_raw_vec();
        let second_array_buf = array_buf.split_off(n * nfeatures);

        let first = Array2::from_shape_vec((n, nfeatures), array_buf).unwrap();
        let second = Array2::from_shape_vec((npoints - n, nfeatures), second_array_buf).unwrap();

        // split targets into two disjoint Vec
        let mut array_buf = self.targets.into_raw_vec();
        let second_array_buf = array_buf.split_off(n);

        let first_targets = Array1::from_shape_vec(n, array_buf).unwrap();
        let second_targets = Array1::from_shape_vec(npoints - n, second_array_buf).unwrap();

        // split weights into two disjoint Vec
        let second_weights = if self.weights.len() == npoints {
            self.weights.split_off(n)
        } else {
            vec![]
        };

        // create new datasets with attached weights
        let dataset1 = Dataset::new(first, first_targets).with_weights(self.weights);
        let dataset2 = Dataset::new(second, second_targets).with_weights(second_weights);
        (dataset1, dataset2)
    }

    /// Performs K-folding on the dataset.
    /// The dataset is divided into `k` "folds", each containing
    /// `(dataset size)/k` samples, used to generate `k` training-validation
    /// dataset pairs. Each pair contains a validation `Dataset` with `k` samples(
    ///  the ones contained in the i-th fold), and a training `Dataset` composed by the
    /// union of all the samples in the remaining folds.
    ///
    /// ### Parameters
    ///
    /// * `k`: the number of folds to apply
    ///
    /// ### Returns
    ///
    /// A vector of `k` training-validation Dataset pairs.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use linfa::dataset::Dataset;
    /// use ndarray::{arr1, arr2};
    ///
    /// let records = arr2(&[[1.,1.], [2.,1.], [3.,2.], [4.,1.],[5., 3.], [6.,2.]]);
    /// let targets = arr1(&[1, 1, 0, 1, 0, 0]);
    ///
    /// let dataset : Dataset<f64, usize> = Dataset::new(records, targets);
    /// let accuracies = dataset.fold(3).into_iter().map(|(train, valid)| {
    ///     // Here you can train your model and perform validation
    ///     
    ///     // let model = params.fit(&dataset);
    ///     // let predi = model.predict(&valid);
    ///     // predi.confusion_matrix(&valid).accuracy()  
    /// });
    /// ```
    ///
    pub fn fold(&self, k: usize) -> Vec<(Dataset<F, E>, Dataset<F, E>)> {
        self.view().fold(k)
    }
}

impl<'a, F: Float, E: Copy> DatasetView<'a, F, E> {
    pub fn bootstrap<R: Rng>(
        &'a self,
        num_samples: usize,
        rng: &'a mut R,
    ) -> impl Iterator<Item = Dataset<F, E>> + 'a {
        std::iter::repeat(()).map(move |_| {
            // sample with replacement
            let indices = (0..num_samples)
                .map(|_| rng.gen_range(0, self.observations()))
                .collect::<Vec<_>>();

            let records = self.records().select(Axis(0), &indices);
            let targets = indices
                .iter()
                .map(|x| self.targets[*x])
                .collect::<ArrayBase<_, Ix1>>();

            Dataset::new(records, targets)
        })
    }

    /// Produces a shuffled version of the current Dataset.
    ///
    /// ### Parameters
    ///
    /// * `rng`: the random number generator that will be used to shuffle the samples
    ///
    /// ### Returns
    ///
    /// A new shuffled version of the current Dataset
    pub fn shuffle<R: Rng>(&self, mut rng: &mut R) -> Dataset<F, E> {
        let mut indices = (0..(&self).observations()).collect::<Vec<_>>();
        indices.shuffle(&mut rng);

        let records = (&self).records().select(Axis(0), &indices);
        let targets = indices
            .iter()
            .map(|x| (self).targets[*x])
            .collect::<Array1<_>>();

        DatasetBase::new(records, targets)
    }

    /// Performs K-folding on the dataset.
    /// The dataset is divided into `k` "fold", each containing
    /// `(dataset size)/k` samples, used to generate `k` training-validation
    /// dataset pairs. Each pair contains a validation `Dataset` with `k` samples,
    ///  the ones contained in the i-th fold, and a training `Dataset` composed by the
    /// union of all the samples in the remaining folds.
    ///
    /// ### Parameters
    ///
    /// * `k`: the number of folds to apply
    ///
    /// ### Returns
    ///
    /// A vector of `k` training-validation Dataset pairs.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use linfa::dataset::DatasetView;
    /// use ndarray::{arr1, arr2};
    ///
    /// let records = arr2(&[[1.,1.], [2.,1.], [3.,2.], [4.,1.],[5., 3.], [6.,2.]]);
    /// let targets = arr1(&[1, 1, 0, 1, 0, 0]);
    ///
    /// let dataset : DatasetView<f64, usize> = DatasetView::new(records.view(), targets.view());
    /// let accuracies = dataset.fold(3).into_iter().map(|(train, valid)| {
    ///     // Here you can train your model and perform validation
    ///     
    ///     // let model = params.fit(&dataset);
    ///     // let predi = model.predict(&valid);
    ///     // predi.confusion_matrix(&valid).accuracy()  
    /// });
    /// ```
    ///  
    pub fn fold(&self, k: usize) -> Vec<(Dataset<F, E>, Dataset<F, E>)> {
        let fold_size = self.targets().dim() / k;

        let mut res = Vec::new();
        for i in 0..k {
            let fold_start = i * fold_size;

            // fold end = max { fold_start + fold_size, #samples}
            let fold_end = if (fold_size * (i + 1)) > self.targets.dim() {
                self.targets.dim()
            } else {
                fold_size * (i + 1)
            };

            let fold_indices = (fold_start..fold_end).collect::<Vec<_>>();
            let non_fold_indices = (0..self.targets.dim())
                .filter(|x| *x < fold_start || *x >= fold_end)
                .collect::<Vec<_>>();

            // remaining records
            let rem_rec = self.records.select(Axis(0), &non_fold_indices);
            // remaining targets
            let rem_tar = self.targets.select(Axis(0), &non_fold_indices);

            // fold records
            let fold_rec = self.records.select(Axis(0), &fold_indices);
            // fold targets
            let fold_tar = self.targets.select(Axis(0), &fold_indices);

            res.push((
                Dataset::new(rem_rec, rem_tar),
                Dataset::new(fold_rec, fold_tar),
            ))
        }
        res
    }
}
