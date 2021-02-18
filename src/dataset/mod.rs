//! Datasets
//!
//! This module implements the dataset struct and various helper traits to extend its
//! functionality.
use ndarray::{
    Array2, ArrayBase, ArrayView, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis,
    CowArray, Ix2, Ix3, NdFloat, OwnedRepr,
};
use num_traits::{FromPrimitive, NumAssignOps, Signed};
use std::cmp::{Ordering, PartialOrd};
use std::collections::HashSet;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::Deref;

use crate::error::{Error, Result};

mod impl_dataset;
mod impl_records;
mod impl_targets;

mod iter;

/// Floating numbers
pub trait Float: NdFloat + FromPrimitive + Signed + Default + Sum + NumAssignOps {}
impl Float for f32 {}
impl Float for f64 {}

/// Discrete labels
pub trait Label: PartialEq + Eq + Hash + Clone {}

impl Label for bool {}
impl Label for usize {}
impl Label for String {}

/// Probability types
///
/// This helper struct exists to distinguish probabilities from floating points. For example SVM
/// selects regression or classification training, based on the target type, and could not
/// distinguish them with floating points alone.
#[derive(Debug, Copy, Clone)]
pub struct Pr(pub f32);

impl PartialEq for Pr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for Pr {
    fn partial_cmp(&self, other: &Pr) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Deref for Pr {
    type Target = f32;

    fn deref(&self) -> &f32 {
        &self.0
    }
}

/// DatasetBase
///
/// A dataset contains a number of records and targets. Each record corresponds to a single target
/// and may be weighted with the `weights` field during the training process.
pub struct DatasetBase<R, T>
where
    R: Records,
{
    pub records: R,
    pub targets: T,

    weights: Vec<f32>,
    feature_names: Vec<String>,
}

/// Targets with precomputed labels
pub struct TargetsWithLabels<L: Label, P> {
    targets: P,
    labels: HashSet<L>,
}

/// Dataset
///
/// The most commonly used typed of dataset. It contains a number of records
/// stored as an `Array2` and each record may correspond to multiple targets. The
/// targets are stored as an `Array2`.
pub type Dataset<D, T> = DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, ArrayBase<OwnedRepr<T>, Ix2>>;

/// DatasetView
///
/// A read only view of a Dataset
pub type DatasetView<'a, D, T> = DatasetBase<ArrayView<'a, D, Ix2>, ArrayView<'a, T, Ix2>>;

/// DatasetPr
///
/// Dataset with probabilities as targets. Useful for multiclass probabilities.
/// It stores records as an `Array2` of elements of type `D`, and targets as an `Array1`
/// of elements of type `Pr`
pub type DatasetPr<D, L> =
    DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, TargetsWithLabels<L, ArrayBase<OwnedRepr<Pr>, Ix3>>>;

/// Records
///
/// The records are input data in the training
pub trait Records: Sized {
    type Elem;

    fn nsamples(&self) -> usize;
    fn nfeatures(&self) -> usize;
}

/// Convert to single or multiple target variables
pub trait AsTargets {
    type Elem;

    /// Returns a view on targets as two-dimensional array
    fn as_multi_targets<'a>(&'a self) -> ArrayView2<'a, Self::Elem>;

    /// Convert to single target, fails for more than one target
    fn try_single_target<'a>(&'a self) -> Result<ArrayView1<'a, Self::Elem>> {
        let multi_targets = self.as_multi_targets();

        if multi_targets.len_of(Axis(1)) > 1 {
            return Err(Error::MultipleTargets);
        }

        Ok(multi_targets.index_axis_move(Axis(1), 0))
    }
}

pub trait FromTargetArray<'a, F> {
    type Owned;
    type View;

    /// Create self object from new target array
    fn new_targets(targets: Array2<F>) -> Self::Owned;
    fn new_targets_view(targets: ArrayView2<'a, F>) -> Self::View;
}

pub trait AsTargetsMut {
    type Elem;

    /// Returns a mutable view on targets as two-dimensional array
    fn as_multi_targets_mut<'a>(&'a mut self) -> ArrayViewMut2<'a, Self::Elem>;

    /// Convert to single target, fails for more than one target
    fn try_single_target_mut<'a>(&'a mut self) -> Result<ArrayViewMut1<'a, Self::Elem>> {
        let multi_targets = self.as_multi_targets_mut();

        if multi_targets.len_of(Axis(1)) > 1 {
            return Err(Error::MultipleTargets);
        }

        Ok(multi_targets.index_axis_move(Axis(1), 0))
    }
}

/// Convert to probability matrix
///
/// Some algorithms are working with probabilities. Targets which allow an implicit conversion into
/// probabilities can implement this trait.
pub trait AsProbabilities {
    fn as_multi_target_probabilities<'a>(&'a self) -> CowArray<'a, Pr, Ix3>;
}

/// Get the labels in all targets
///
pub trait Labels {
    type Elem: Label;

    fn label_set(&self) -> HashSet<Self::Elem>;
    fn labels(&self) -> Vec<Self::Elem> {
        self.label_set().into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use rand::{rngs::SmallRng, SeedableRng};

    #[test]
    fn dataset_implements_required_methods() {
        let mut rng = SmallRng::seed_from_u64(42);

        // ------ Targets ------

        // New
        let mut dataset = Dataset::new(array![[1., 2.], [1., 2.]], array![0., 1.]);

        // Shuffle
        dataset = dataset.shuffle(&mut rng);

        // Bootstrap
        let mut iter = dataset.bootstrap(3, &mut rng);
        for _ in 1..5 {
            let b_dataset = iter.next().unwrap();
            assert_eq!(b_dataset.records().dim().0, 3);
        }

        let linspace: Array1<f64> = Array1::linspace(0.0, 0.8, 100);
        let records = Array2::from_shape_vec((50, 2), linspace.to_vec()).unwrap();
        let targets: Array1<f64> = Array1::linspace(0.0, 0.8, 50);
        let dataset = Dataset::new(records, targets);

        //Split with ratio view
        let (train, val) = dataset.split_with_ratio_view(0.5);
        assert_eq!(train.targets().dim(), 25);
        assert_eq!(val.targets().dim(), 25);
        assert_eq!(train.records().dim().0, 25);
        assert_eq!(val.records().dim().0, 25);

        // Split with ratio
        let (train, val) = dataset.split_with_ratio(0.25);
        assert_eq!(train.targets().dim(), 13);
        assert_eq!(val.targets().dim(), 37);
        assert_eq!(train.records().dim().0, 13);
        assert_eq!(val.records().dim().0, 37);

        // ------ Labels ------
        let dataset_multiclass =
            Dataset::new(array![[1., 2.], [2., 1.], [0., 0.]], array![0, 1, 2]);

        // One Vs All
        let datasets_one_vs_all = dataset_multiclass.one_vs_all();
        assert_eq!(datasets_one_vs_all.len(), 3);

        for dataset in datasets_one_vs_all.iter() {
            assert_eq!(dataset.labels().iter().filter(|x| **x).count(), 1);
        }

        let dataset_multiclass = Dataset::new(
            array![[1., 2.], [2., 1.], [0., 0.], [2., 2.]],
            array![0, 1, 2, 2],
        );

        // Frequencies with mask
        let freqs = dataset_multiclass.frequencies_with_mask(&[true, true, true, true]);
        assert_eq!(*freqs.get(&0).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&1).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&2).unwrap() as usize, 2);

        let freqs = dataset_multiclass.frequencies_with_mask(&[true, true, true, false]);
        assert_eq!(*freqs.get(&0).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&1).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&2).unwrap() as usize, 1);
    }

    #[test]
    fn dataset_view_implements_required_methods() {
        let mut rng = SmallRng::seed_from_u64(42);
        let observations = array![[1., 2.], [1., 2.]];
        let targets = array![0., 1.];

        // ------ Targets ------

        // New
        let dataset_view = DatasetView::new(observations.view(), targets.view());

        // Shuffle
        let _shuffled_owned = dataset_view.shuffle(&mut rng);

        // Bootstrap
        let mut iter = dataset_view.bootstrap(3, &mut rng);
        for _ in 1..5 {
            let b_dataset = iter.next().unwrap();
            assert_eq!(b_dataset.records().dim().0, 3);
        }

        let linspace: Array1<f64> = Array1::linspace(0.0, 0.8, 100);
        let records = Array2::from_shape_vec((50, 2), linspace.to_vec()).unwrap();
        let targets: Array1<f64> = Array1::linspace(0.0, 0.8, 50);
        let dataset = Dataset::new(records, targets);

        // view ,Split with ratio view
        let view: DatasetView<f64, f64> = dataset.view();

        let (train, val) = view.split_with_ratio_view(0.5);
        assert_eq!(train.targets().dim(), 25);
        assert_eq!(val.targets().dim(), 25);
        assert_eq!(train.records().dim().0, 25);
        assert_eq!(val.records().dim().0, 25);

        // ------ Labels ------
        let dataset_multiclass =
            Dataset::new(array![[1., 2.], [2., 1.], [0., 0.]], array![0, 1, 2]);
        let view: DatasetView<f64, usize> = dataset_multiclass.view();

        // One Vs All
        let datasets_one_vs_all = view.one_vs_all();
        assert_eq!(datasets_one_vs_all.len(), 3);

        for dataset in datasets_one_vs_all.iter() {
            assert_eq!(dataset.labels().iter().filter(|x| **x).count(), 1);
        }

        let dataset_multiclass = Dataset::new(
            array![[1., 2.], [2., 1.], [0., 0.], [2., 2.]],
            array![0, 1, 2, 2],
        );
        let view: DatasetView<f64, usize> = dataset_multiclass.view();

        // Frequencies with mask
        let freqs = view.frequencies_with_mask(&[true, true, true, true]);
        assert_eq!(*freqs.get(&0).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&1).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&2).unwrap() as usize, 2);

        let freqs = view.frequencies_with_mask(&[true, true, true, false]);
        assert_eq!(*freqs.get(&0).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&1).unwrap() as usize, 1);
        assert_eq!(*freqs.get(&2).unwrap() as usize, 1);
    }

    #[test]
    fn datasets_have_k_fold() {
        let linspace: Array1<f64> = Array1::linspace(0.0, 0.8, 100);
        let records = Array2::from_shape_vec((50, 2), linspace.to_vec()).unwrap();
        let targets: Array1<f64> = Array1::linspace(0.0, 0.8, 50);
        for (train, val) in DatasetView::new(records.view(), targets.view())
            .fold(2)
            .into_iter()
        {
            assert_eq!(train.records().dim(), (25, 2));
            assert_eq!(val.records().dim(), (25, 2));
            assert_eq!(train.targets().dim(), 25);
            assert_eq!(val.targets().dim(), 25);
        }
        assert_eq!(Dataset::new(records, targets).fold(10).len(), 10);

        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        for (i, (train, val)) in Dataset::new(records, targets)
            .fold(5)
            .into_iter()
            .enumerate()
        {
            assert_eq!(val.records.row(0)[0] as usize, (i + 1));
            assert_eq!(val.records.row(0)[1] as usize, (i + 1));
            assert_eq!(val.targets[0] as usize, (i + 1));

            for j in 0..4 {
                assert!(train.records.row(j)[0] as usize != (i + 1));
                assert!(train.records.row(j)[1] as usize != (i + 1));
                assert!(train.targets[j] as usize != (i + 1));
            }
        }
    }

    struct MockFittable {}

    struct MockFittableResult {
        mock_var: usize,
    }

    use crate::traits::Fit;
    use ndarray::{ArrayView1, ArrayView2};

    impl<'a> Fit<'a, ArrayView2<'a, f64>, ArrayView1<'a, f64>> for MockFittable {
        type Object = MockFittableResult;

        fn fit(&self, training_data: &DatasetView<f64, f64>) -> Self::Object {
            MockFittableResult {
                mock_var: training_data.targets().dim(),
            }
        }
    }

    #[test]
    fn test_iter_fold() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = MockFittable {};

        for (i, (model, validation_set)) in dataset.iter_fold(5, |v| params.fit(&v)).enumerate() {
            assert_eq!(model.mock_var, 4);
            assert_eq!(validation_set.records().row(0)[0] as usize, i + 1);
            assert_eq!(validation_set.records().row(0)[1] as usize, i + 1);
            assert_eq!(validation_set.targets()[0] as usize, i + 1);
            assert_eq!(validation_set.records().dim(), (1, 2));
            assert_eq!(validation_set.targets().dim(), 1);
        }
    }

    #[test]
    fn test_iter_fold_uneven_folds() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = MockFittable {};

        // If we request three folds from a dataset with 5 samples it will cut the
        // last two samples from the folds and always add them as a tail of the training
        // data
        for (i, (model, validation_set)) in dataset.iter_fold(3, |v| params.fit(&v)).enumerate() {
            assert_eq!(model.mock_var, 4);
            assert_eq!(validation_set.records().row(0)[0] as usize, i + 1);
            assert_eq!(validation_set.records().row(0)[1] as usize, i + 1);
            assert_eq!(validation_set.targets()[0] as usize, i + 1);
            assert_eq!(validation_set.records().dim(), (1, 2));
            assert_eq!(validation_set.targets().dim(), 1);
            assert!(i < 3);
        }

        // the same goes for the last sample if we choose 4 folds
        for (i, (model, validation_set)) in dataset.iter_fold(4, |v| params.fit(&v)).enumerate() {
            assert_eq!(model.mock_var, 4);
            assert_eq!(validation_set.records().row(0)[0] as usize, i + 1);
            assert_eq!(validation_set.records().row(0)[1] as usize, i + 1);
            assert_eq!(validation_set.targets()[0] as usize, i + 1);
            assert_eq!(validation_set.records().dim(), (1, 2));
            assert_eq!(validation_set.targets().dim(), 1);
            assert!(i < 4);
        }

        // if we choose 2 folds then again the last sample will be only
        // used for trainig
        for (i, (model, validation_set)) in dataset.iter_fold(2, |v| params.fit(&v)).enumerate() {
            assert_eq!(model.mock_var, 3);
            assert_eq!(validation_set.targets().dim(), 2);
            assert!(i < 2);
        }
    }

    #[test]
    #[should_panic]
    fn iter_fold_panics_k_0() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = MockFittable {};
        let _ = dataset.iter_fold(0, |v| params.fit(&v)).enumerate();
    }

    #[test]
    #[should_panic]
    fn iter_fold_panics_k_more_than_samples() {
        let records =
            Array2::from_shape_vec((5, 2), vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]).unwrap();
        let targets = Array1::from_shape_vec(5, vec![1., 2., 3., 4., 5.]).unwrap();
        let mut dataset: Dataset<f64, f64> = (records, targets).into();
        let params = MockFittable {};
        let _ = dataset.iter_fold(6, |v| params.fit(&v)).enumerate();
    }
}
