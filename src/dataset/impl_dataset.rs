use ndarray::{
    stack, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Dimension, Ix2, ArrayViewMut2, DataMut
};
use rand::{seq::SliceRandom, Rng};
use std::collections::{HashMap, HashSet};

use super::{
    iter::{ChunksIter, Iter}, Dataset, DatasetBase, DatasetView, Float, Label, Records, AsTargets, AsTargetsMut, Result, Labels, FromTargetArray
};

/// Iterate over observations
///
/// This function creates an iterator which produces tuples of data points and target value. The
/// iterator runs once for each data point and, while doing so, holds an reference to the owned
/// dataset.
///
/// # Example
/// ```
/// let dataset = linfa_datasets::iris();
///
/// for (x, y) in dataset.iter() {
///     println!("{} => {}", x, y);
/// }
/// ```
impl<F: Float, L: Label> DatasetBase<Array2<F>, Array1<L>> {
    pub fn iter(&self) -> Iter<'_, Array2<F>, Array1<L>> {
        Iter::new(&self.records, &self.targets)
    }
}

/// Implementation without constraints on records and targets
///
/// This implementation block provides methods for the creation and mutation of datasets. This
/// includes swapping the targets, return the records etc.
impl<R: Records, S> DatasetBase<R, S> {
    /// Create a new dataset from records and targets
    ///
    /// # Example
    ///
    /// ```ignore
    /// let dataset = Dataset::new(records, targets);
    /// ```
    pub fn new(records: R, targets: S) -> DatasetBase<R, S> {
        DatasetBase {
            records,
            targets,
            weights: Vec::new(),
            feature_names: Vec::new(),
        }
    }

    /// Returns reference to targets
    pub fn targets(&self) -> &S {
        &self.targets
    }

    /// Returns optionally weights
    pub fn weights(&self) -> Option<&[f32]> {
        if !self.weights.is_empty() {
            Some(&self.weights)
        } else {
            None
        }
    }

    /// Return a single weight
    ///
    /// The weight of the `idx`th observation is returned. If no weight is specified, then all
    /// observations are unweighted with default value `1.0`.
    pub fn weight_for(&self, idx: usize) -> f32 {
        self.weights.get(idx).copied().unwrap_or(1.0)
    }

    /// Returns feature names
    ///
    /// A feature name gives a human-readable string describing the purpose of a single feature.
    /// This allow the reader to understand its purpose while analysing results, for example
    /// correlation analysis or feature importance.
    pub fn feature_names(&self) -> Vec<String> {
        if !self.feature_names.is_empty() {
            self.feature_names.clone()
        } else {
            (0..self.records.nfeatures())
                .map(|idx| format!("feature-{}", idx))
                .collect()
        }
    }

    /// Return records of a dataset
    ///
    /// The records are data points from which predictions are made. This functions returns a
    /// reference to the record field.
    pub fn records(&self) -> &R {
        &self.records
    }

    /// Updates the records of a dataset
    ///
    /// This function overwrites the records in a dataset. It also invalidates the weights and
    /// feature names.
    pub fn with_records<T: Records>(self, records: T) -> DatasetBase<T, S> {
        DatasetBase {
            records,
            targets: self.targets,
            weights: Vec::new(),
            feature_names: Vec::new(),
        }
    }

    /// Updates the targets of a dataset
    ///
    /// This function overwrites the targets in a dataset.
    pub fn with_targets<T>(self, targets: T) -> DatasetBase<R, T> {
        DatasetBase {
            records: self.records,
            targets,
            weights: self.weights,
            feature_names: self.feature_names,
        }
    }

    /// Updates the weights of a dataset
    pub fn with_weights(mut self, weights: Vec<f32>) -> DatasetBase<R, S> {
        self.weights = weights;

        self
    }

    /// Updates the feature names of a dataset
    pub fn with_feature_names<I: Into<String>>(mut self, names: Vec<I>) -> DatasetBase<R, S> {
        let feature_names = names.into_iter().map(|x| x.into()).collect();

        self.feature_names = feature_names;

        self
    }
}

/// Map targets with a function `f`
///
/// # Example
///
/// ```
/// let dataset = linfa_datasets::winequality()
///     .map_targets(|x| *x > 6);
///
/// // dataset has now boolean targets
/// println!("{:?}", dataset.targets());
/// ```
///
/// # Returns
///
/// A modified dataset with new target type.
///
impl<L, R: Records, T: AsTargets<Elem = L>> DatasetBase<R, T> {
    pub fn map_targets<S, G: FnMut(&L) -> S>(self, fnc: G) -> DatasetBase<R, Array2<S>> {
        let DatasetBase {
            records,
            targets,
            weights,
            feature_names,
            ..
        } = self;

        let targets = targets.as_multi_targets();

        DatasetBase {
            records,
            targets: targets.map(fnc),
            weights,
            feature_names,
        }
    }

    pub fn ntargets(&self) -> usize {
        self.targets.as_multi_targets().len_of(Axis(1))
    }
}

impl<L, R: Records, T: AsTargets<Elem = L>> AsTargets for DatasetBase<R, T> {
    type Elem = L;

    fn as_multi_targets<'a>(&'a self) -> ArrayView2<'a, Self::Elem> {
        self.targets.as_multi_targets()
    }
}

impl<L, R: Records, T: AsTargetsMut<Elem = L>> AsTargetsMut for DatasetBase<R, T> {
    type Elem = L;

    fn as_multi_targets_mut<'a>(&'a mut self) -> ArrayViewMut2<'a, Self::Elem> {
        self.targets.as_multi_targets_mut()
    }
}

impl<'a, L: 'a, F: Float, T> DatasetBase<ArrayView2<'a, F>, T> where
    T: AsTargets<Elem = L> + FromTargetArray<'a, L> {

    /// Split dataset into two disjoint chunks
    ///
    /// This function splits the observations in a dataset into two disjoint chunks. The splitting
    /// threshold is calculated with the `ratio`. For example a ratio of `0.9` allocates 90% to the
    /// first chunks and 9% to the second. This is often used in training, validation splitting
    /// procedures.
    pub fn split_with_ratio(&'a self, ratio: f32,) -> (DatasetBase<ArrayView2<'a, F>, T::View>, DatasetBase<ArrayView2<'a, F>, T::View>) {
        let n = (self.nsamples() as f32 * ratio).ceil() as usize;
        let (records_first, records_second) = self.records.view().split_at(Axis(0), n);
        let (targets_first, targets_second) = self.as_multi_targets().split_at(Axis(0), n);

        let targets_first = T::new_targets_view(targets_first);
        let targets_second = T::new_targets_view(targets_second);

        (
            DatasetBase::new(records_first, targets_first),
            DatasetBase::new(records_second, targets_second)
        )
    }

    /// Creates a view of a dataset
    pub fn view(&'a self) -> DatasetBase<ArrayView2<'a, F>, T::View> {
        let records = self.records().view();
        let targets = T::new_targets_view(self.as_multi_targets());
        DatasetBase::new(records, targets)
    }
}

impl<L: Label, T: Labels<Elem = L>, R: Records> Labels for DatasetBase<R, T> {
    type Elem = L;

    fn label_set(&self) -> HashSet<L> {
        self.targets().label_set()
    }
}

impl<'a, 'b: 'a, F: Float, L: Label, T, D> DatasetBase<ArrayBase<D, Ix2>, T> where 
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L> + FromTargetArray<'a, bool>,
{
    /// Produce N boolean targets from multi-class targets
    ///
    /// Some algorithms (like SVM) don't support multi-class targets. This function splits a
    /// dataset into multiple binary target view of the same dataset.
    pub fn one_vs_all(&self) -> Result<Vec<DatasetBase<ArrayView2<'_, F>, T::Owned>>> {
        let targets = self.targets().try_single_target()?;

        Ok(self.labels()
            .into_iter()
            .map(|label| {
                let targets = targets
                    .iter()
                    .map(|x| x == &label)
                    .collect::<Array1<_>>()
                    .insert_axis(Axis(1));

                DatasetBase::new(self.records().view(), T::new_targets(targets))
            })
            .collect())
    }
}

impl<L: Label, R: Records, S: AsTargets<Elem = L>> DatasetBase<R, S> {
    /// Calculates label frequencies from a dataset while masking certain samples.
    ///
    /// ### Parameters
    ///
    /// * `mask`: a boolean array that specifies which samples to include in the count
    ///
    /// ### Returns
    ///
    /// A mapping of the Dataset's samples to their frequencies
    pub fn frequencies_with_mask<'a>(&'a self, mask: &[bool]) -> HashMap<L, f32> {
        let mut freqs = HashMap::new();

        for (elms, val) in self
            .targets
            .as_multi_targets()
            .axis_iter(Axis(0))
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(i, x)| (x, self.weight_for(i)))
        {
            for elm in elms {
                if !freqs.contains_key(elm) {
                    freqs.insert(elm.clone(), 0.0);
                }

                *freqs.get_mut(&elm).unwrap() += val;
            }
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
            feature_names: Vec::new(),
        }
    }
}

impl<F: Float, T: AsTargets, D: Data<Elem = F>, I: Dimension> From<(ArrayBase<D, I>, T)>
    for DatasetBase<ArrayBase<D, I>, T>
{
    fn from(rec_tar: (ArrayBase<D, I>, T)) -> Self {
        DatasetBase {
            records: rec_tar.0,
            targets: rec_tar.1,
            weights: Vec::new(),
            feature_names: Vec::new(),
        }
    }
}

impl<'b, F: Float, E: Copy + 'b, D, T> DatasetBase<ArrayBase<D, Ix2>, T> where
    D: Data<Elem = F>,
    T: AsTargets<Elem = E> + FromTargetArray<'b, E>,
    T::Owned: AsTargets,
{
    /// Apply bootstrapping for samples and features
    ///
    /// Bootstrap aggregating is used for sub-sample generation and improves the accuracy and
    /// stability of machine learning algorithms. It samples data uniformly with replacement and
    /// generates datasets where elements may be shared. This selects a subset of observations as
    /// well as features.
    ///
    /// # Parameters
    ///
    ///  * `sample_feature_size`: The number of samples and features per bootstrap
    ///  * `rng`: The random number generator used in the sampling procedure
    ///
    ///  # Returns
    ///
    ///  An infinite Iterator yielding at each step a new bootstrapped dataset
    ///
    pub fn bootstrap<R: Rng>(
        &'b self,
        sample_feature_size: (usize, usize),
        rng: &'b mut R,
    ) -> impl Iterator<Item = DatasetBase<Array2<F>, <T as FromTargetArray<'b, E>>::Owned>> + 'b {
        std::iter::repeat(()).map(move |_| {
            // sample with replacement
            let indices = (0..sample_feature_size.0)
                .map(|_| rng.gen_range(0, self.nsamples()))
                .collect::<Vec<_>>();

            let records = self.records().select(Axis(0), &indices);
            let targets = T::new_targets(self.as_multi_targets().select(Axis(0), &indices));

            let indices = (0..sample_feature_size.1)
                .map(|_| rng.gen_range(0, self.nfeatures()))
                .collect::<Vec<_>>();

            let records = records.select(Axis(1), &indices);
                    
            DatasetBase::new(records, targets)
        })
    }

    /// Apply sample bootstrapping
    ///
    /// Bootstrap aggregating is used for sub-sample generation and improves the accuracy and
    /// stability of machine learning algorithms. It samples data uniformly with replacement and
    /// generates datasets where elements may be shared. Only a sample subset is selected which
    /// retains all features and targets.
    ///
    /// # Parameters
    ///
    ///  * `num_samples`: The number of samples per bootstrap
    ///  * `rng`: The random number generator used in the sampling procedure
    ///
    ///  # Returns
    ///
    ///  An infinite Iterator yielding at each step a new bootstrapped dataset
    ///
    pub fn bootstrap_samples<R: Rng>(
        &'b self,
        num_samples: usize,
        rng: &'b mut R,
    ) -> impl Iterator<Item = DatasetBase<Array2<F>, <T as FromTargetArray<'b, E>>::Owned>> + 'b {
        std::iter::repeat(()).map(move |_| {
            // sample with replacement
            let indices = (0..num_samples)
                .map(|_| rng.gen_range(0, self.nsamples()))
                .collect::<Vec<_>>();

            let records = self.records().select(Axis(0), &indices);
            let targets = T::new_targets(self.as_multi_targets().select(Axis(0), &indices));

            DatasetBase::new(records, targets)
        })
    }

    /// Apply feature bootstrapping
    ///
    /// Bootstrap aggregating is used for sub-sample generation and improves the accuracy and
    /// stability of machine learning algorithms. It samples data uniformly with replacement and
    /// generates datasets where elements may be shared. Only a feature subset is selected while
    /// retaining all samples and targets.
    ///
    /// # Parameters
    ///
    ///  * `num_features`: The number of features per bootstrap
    ///  * `rng`: The random number generator used in the sampling procedure
    ///
    ///  # Returns
    ///
    ///  An infinite Iterator yielding at each step a new bootstrapped dataset
    ///
    pub fn bootstrap_features<R: Rng>(
        &'b self,
        num_features: usize,
        rng: &'b mut R,
    ) -> impl Iterator<Item = DatasetBase<Array2<F>, <T as FromTargetArray<'b, E>>::Owned>> + 'b {
        std::iter::repeat(()).map(move |_| {
            let targets = T::new_targets(self.as_multi_targets().to_owned());

            let indices = (0..num_features)
                .map(|_| rng.gen_range(0, self.nfeatures()))
                .collect::<Vec<_>>();

            let records = self.records.select(Axis(1), &indices);

            DatasetBase::new(records, targets)
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
    pub fn shuffle<R: Rng>(self, rng: &mut R) -> DatasetBase<Array2<F>, T::Owned> {
        let mut indices = (0..self.nsamples()).collect::<Vec<_>>();
        indices.shuffle(rng);

        let records = (&self).records().select(Axis(0), &indices);
        let targets = (&self).as_multi_targets().select(Axis(0), &indices);
        let targets = T::new_targets(targets);

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
    pub fn fold(&self, k: usize) -> Vec<(DatasetBase<Array2<F>, T::Owned>, DatasetBase<Array2<F>, T::Owned>)> {
        let targets = self.as_multi_targets();
        let fold_size = targets.len() / k;
        let mut res = Vec::new();

        // Generates all k folds of records and targets
        let mut records_chunks: Vec<_> =
            self.records.axis_chunks_iter(Axis(0), fold_size).collect();
        let mut targets_chunks: Vec<_> =
            targets.axis_chunks_iter(Axis(0), fold_size).collect();

        // For each iteration, take the first chunk for both records and targets as the validation set and
        // stack all the other chunks to create the training set. In the end swap the first chunk with the
        // one in the next index so that it is ready for the next iteration
        for i in 0..k {
            let remaining_records = stack(Axis(0), &records_chunks.as_slice()[1..]).unwrap();
            let remaining_targets = stack(Axis(0), &targets_chunks.as_slice()[1..]).unwrap();

            res.push((
                // training
                DatasetBase::new(remaining_records, T::new_targets(remaining_targets)),
                // validation
                DatasetBase::new(
                    records_chunks[0].into_owned(),
                    T::new_targets(targets_chunks[0].into_owned()),
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

    pub fn sample_chunks<'a: 'b>(
        &'b self,
        chunk_size: usize,
    ) -> ChunksIter<'b, 'a, F, T> {
        ChunksIter::new(self.records().view(), &self.targets, chunk_size, Axis(0))
    }

    pub fn to_owned(&self) -> DatasetBase<Array2<F>, T::Owned> {
        (self.records().to_owned(), T::new_targets(self.as_multi_targets().to_owned())).into()
    }
}

impl<'b, F: Float, E: Copy + 'b, D, T> DatasetBase<ArrayBase<D, Ix2>, T> where
    D: DataMut<Elem = F>,
    T: AsTargets<Elem = E> + AsTargetsMut<Elem = E> + FromTargetArray<'b, E>,
    T::Owned: AsTargets,
{

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
    pub fn iter_fold<'a: 'b, O: 'a, C: 'a + Fn(DatasetBase<ArrayView2<'a, F>, <T as FromTargetArray<'b, E>>::View>) -> O>(
        &'b mut self,
        k: usize,
        fit_closure: C,
    ) -> impl Iterator<Item = (O, DatasetBase<ArrayView2<'b, F>, <T as FromTargetArray<'b, E>>::View>)> + 'b {
        assert!(k > 0);
        assert!(k <= self.nsamples());
        let samples_count = self.nsamples();
        let fold_size = samples_count / k;

        let features = self.nfeatures();
        let targets = self.ntargets();

        let mut objs: Vec<O> = Vec::new();

        for i in 0..k {
            {
                let mut records_sl = self.records.view_mut();
                assist_swap_array22(records_sl, i, fold_size, features);
            }
            {
                let mut targets_sl = self.targets.as_multi_targets_mut();
                assist_swap_array22(targets_sl, i, fold_size, targets);
            }

            {
            /*let records_sl = self.records.view();
            let targets_sl = self.as_multi_targets();
            let (train, valid) = targets_sl.split_at(Axis(0), samples_count - fold_size);
            let (train_r, valid_r) = records_sl.split_at(Axis(0), fold_size);

            let train = DatasetBase::new(
                train_r,
                T::new_targets_view(
                    train
                )
            );

            let obj = fit_closure(train);
            objs.push(obj);*/
            }

            {
                let mut records_sl = self.records.view_mut();
                assist_swap_array22(records_sl, i, fold_size, features);
            }
            {
                let mut targets_sl = self.targets.as_multi_targets_mut();
                assist_swap_array22(targets_sl, i, fold_size, targets);
            }
        }

        objs.into_iter()
            .zip(self.sample_chunks(fold_size))
    }
}

/*fn assist_swap_array1<E>(slice: &mut [E], index: usize, fold_size: usize) {
    if index == 0 {
        return;
    }
    let start = fold_size * index;
    let (first_s, second_s) = slice.split_at_mut(start);
    let (mut fold, _) = second_s.split_at_mut(fold_size);
    first_s[..fold_size].swap_with_slice(&mut fold);
}*/

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

fn assist_swap_array22<F>(mut slice: ArrayViewMut2<F>, index: usize, fold_size: usize, features: usize) {
    let slice = slice.as_slice_mut().unwrap();

    if index == 0 {
        return;
    }
    let adj_fold_size = fold_size * features;
    let start = adj_fold_size * index;
    let (first_s, second_s) = slice.split_at_mut(start);
    let (mut fold, _) = second_s.split_at_mut(adj_fold_size);
    first_s[..fold_size * features].swap_with_slice(&mut fold);
}
impl <F: Float, E> Dataset<F, E> {
    /// Split dataset into two disjoint chunks
    ///
    /// This function splits the observations in a dataset into two disjoint chunks. The splitting
    /// threshold is calculated with the `ratio`. If the input Dataset contains `n` samples then the
    /// two new Datasets will have respectively `n * ratio` and `n - (n*ratio)` samples.
    /// For example a ratio of `0.9` allocates 90% to the
    /// first chunks and 10% to the second. This is often used in training, validation splitting
    /// procedures.
    ///
    /// ### Parameters
    ///
    /// * `ratio`: the ratio of samples in the input Dataset to include in the first output one
    ///
    /// ### Returns
    ///  
    /// The input Dataset split into two according to the input ratio.
    pub fn split_with_ratio(mut self, ratio: f32) -> (Self, Self) {
        let (nfeatures, ntargets) = (
            self.nfeatures(),
            self.ntargets()
        );
            
        let n1 = (self.nsamples() as f32 * ratio).ceil() as usize;
        let n2 = self.nsamples() - n1;

        // split records into two disjoint arrays
        let mut array_buf = self.records.into_raw_vec();
        let second_array_buf = array_buf.split_off(n1 * nfeatures);

        let first = Array2::from_shape_vec((n1, nfeatures), array_buf).unwrap();
        let second = Array2::from_shape_vec((n2, nfeatures), second_array_buf).unwrap();

        // split targets into two disjoint Vec
        let mut array_buf = self.targets.into_raw_vec();
        let second_array_buf = array_buf.split_off(n1 * ntargets);

        let first_targets = Array2::from_shape_vec((n1, ntargets), array_buf).unwrap();
        let second_targets = Array2::from_shape_vec((n2, ntargets), second_array_buf).unwrap();

        // split weights into two disjoint Vec
        let second_weights = if self.weights.len() == n1+n2 {
            self.weights.split_off(n1)
        } else {
            vec![]
        };

        // create new datasets with attached weights
        let dataset1 = Dataset::new(first, first_targets).with_weights(self.weights);
        let dataset2 = Dataset::new(second, second_targets).with_weights(second_weights);
        (dataset1, dataset2)
    }
}

