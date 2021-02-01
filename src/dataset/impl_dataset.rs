use ndarray::{
    stack, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Dimension, Ix1, Ix2,
};
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

    pub fn axis_chunks_iter(
        &self,
        axis: Axis,
        chunk_size: usize,
    ) -> impl Iterator<Item = DatasetView<F, E>> {
        self.records()
            .axis_chunks_iter(axis, chunk_size)
            .zip(self.targets().axis_chunks_iter(axis, chunk_size))
            .map(|(rec, tar)| (rec, tar).into())
    }

    /// Allows to perform k-folding cross validation on fittable algorithms.
    ///
    /// Given in input a dataset, a value of k and the desired params for the fittable
    /// algorithm, returns an iterator over the k trained models and the
    /// associated validation set.
    ///
    /// The models are trained according to a closure specified
    /// as an input.
    ///
    /// ## Parameters
    ///
    /// - `k`: the number of folds to apply to the dataset
    /// - `params`: the desired parameters for the fittable algorithm at hand
    /// - `fit_closure`: a closure of the type `(params, training_data) -> fitted_model`
    /// that will be used to produce the trained model for each fold. The training data given in input
    /// won't outlive the closure.
    ///
    /// ## Returns
    ///
    /// An iterator over couples `(trained_model, validation_set)`.
    ///
    /// ## Panics
    ///
    /// This method will panic for any of the following three reasons:
    ///
    /// - The value of `k` provided is not positive;
    /// - The value of `k` provided is greater than the total number of samples in the dataset;
    /// - The dataset's data is not stored contiguously and in standard order;
    ///
    /// ## Example
    /// ```rust
    /// use linfa::traits::Fit;
    /// use linfa::dataset::{Dataset, DatasetView};
    /// use ndarray::{array, ArrayView1, ArrayView2};
    ///
    /// struct MockFittable {}
    ///
    /// struct MockFittableResult {
    ///     mock_var: usize,
    /// }
    ///
    /// impl<'a> Fit<'a, ArrayView2<'a, f64>, ArrayView1<'a, f64>> for MockFittable {
    ///     type Object = MockFittableResult;
    ///
    ///     fn fit(&self, training_data: &DatasetView<f64, f64>) -> Self::Object {
    ///         MockFittableResult { mock_var: training_data.targets().dim()}
    ///     }
    /// }
    ///
    /// let records = array![[1.,1.], [2.,2.], [3.,3.], [4.,4.], [5.,5.]];
    /// let targets = array![1.,2.,3.,4.,5.];
    /// let mut dataset: Dataset<f64, f64> = (records, targets).into();
    /// let params = MockFittable {};
    ///
    ///for (model,validation_set) in dataset.iter_fold(5, |v| params.fit(&v)){
    ///     // Here you can use `model` and `validation_set` to
    ///     // assert the performance of the chosen algorithm
    /// }
    /// ```
    pub fn iter_fold<'a, O: 'a, C: 'a + Fn(DatasetView<F, E>) -> O>(
        &'a mut self,
        k: usize,
        fit_closure: C,
    ) -> impl Iterator<Item = (O, DatasetView<F, E>)> + 'a {
        assert!(k > 0);
        assert!(k <= self.targets.len());
        let samples_count = self.targets().len();
        let fold_size = samples_count / k;

        let features = self.records.dim().1;

        let mut records_sl = self.records.as_slice_mut().unwrap();
        let mut targets_sl = self.targets.as_slice_mut().unwrap();

        let mut objs: Vec<O> = Vec::new();

        for i in 0..k {
            assist_swap_array2(&mut records_sl, i, fold_size, features);
            assist_swap_array1(&mut targets_sl, i, fold_size);

            let train = DatasetView::new(
                ArrayView2::from_shape(
                    (samples_count - fold_size, features),
                    records_sl.split_at(fold_size * features).1,
                )
                .unwrap(),
                ArrayView1::from_shape(samples_count - fold_size, targets_sl.split_at(fold_size).1)
                    .unwrap(),
            );

            let obj = fit_closure(train);
            objs.push(obj);

            assist_swap_array2(&mut records_sl, i, fold_size, features);
            assist_swap_array1(&mut targets_sl, i, fold_size);
        }
        objs.into_iter()
            .zip(self.axis_chunks_iter(Axis(0), fold_size))
    }
}

fn assist_swap_array1<E>(slice: &mut [E], index: usize, fold_size: usize) {
    if index == 0 {
        return;
    }
    let start = fold_size * index;
    let (first_s, second_s) = slice.split_at_mut(start);
    let (mut fold, _) = second_s.split_at_mut(fold_size);
    first_s[..fold_size].swap_with_slice(&mut fold);
}

fn assist_swap_array2<F>(slice: &mut [F], index: usize, fold_size: usize, features: usize) {
    if index == 0 {
        return;
    }
    let adj_fold_size = fold_size * features;
    let start = adj_fold_size * index;
    let (first_s, second_s) = slice.split_at_mut(start);
    let (mut fold, _) = second_s.split_at_mut(adj_fold_size);
    first_s[..fold_size * features].swap_with_slice(&mut fold);
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
        let fold_size = self.targets().len() / k;
        let mut res = Vec::new();

        // Generates all k folds of records and targets
        let mut records_chunks: Vec<_> =
            self.records.axis_chunks_iter(Axis(0), fold_size).collect();
        let mut targets_chunks: Vec<_> =
            self.targets.axis_chunks_iter(Axis(0), fold_size).collect();

        // For each iteration, take the first chunk for both records and targets as the validation set and
        // stack all the other chunks to create the training set. In the end swap the first chunk with the
        // one in the next index so that it is ready for the next iteration
        for i in 0..k {
            let remaining_records = stack(Axis(0), &records_chunks.as_slice()[1..]).unwrap();
            let remaining_targets = stack(Axis(0), &targets_chunks.as_slice()[1..]).unwrap();

            res.push((
                // training
                Dataset::new(remaining_records, remaining_targets),
                // validation
                Dataset::new(
                    records_chunks[0].into_owned(),
                    targets_chunks[0].into_owned(),
                ),
            ));

            // swap
            if i < k - 1 {
                records_chunks.swap(0, i + 1);
                targets_chunks.swap(0, i + 1);
            }
        }
        res
    }

    pub fn to_owned(&self) -> Dataset<F, E> {
        (self.records().to_owned(), self.targets.to_owned()).into()
    }
}
