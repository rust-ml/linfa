//! Datasets
//!
//! This module implements the dataset struct and various helper traits to extend its
//! functionality.
use ndarray::{ArrayBase, ArrayView, Ix1, Ix2, NdFloat, OwnedRepr};
use num_traits::{FromPrimitive, Signed};
use std::cmp::{Ordering, PartialOrd};
use std::hash::Hash;
use std::iter::Sum;
use std::ops::Deref;

mod impl_dataset;
mod impl_records;
mod impl_targets;

mod iter;

/// Floating numbers
pub trait Float: NdFloat + FromPrimitive + Signed + Default + Sum {}
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
    T: Targets,
{
    pub records: R,
    pub targets: T,

    weights: Vec<f32>,
}

/// Dataset
///
/// The most commonly used typed of dataset. It contains a number of records
/// stored as an `Array2` and each record corresponds to a single target. Such
/// targets are stored as an `Array1`.
pub type Dataset<D, T> = DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, ArrayBase<OwnedRepr<T>, Ix1>>;

/// DatasetView
///
/// A read only view of a Dataset
pub type DatasetView<'a, D, T> = DatasetBase<ArrayView<'a, D, Ix2>, ArrayView<'a, T, Ix1>>;

/// DatasetPr
///
/// Dataset with probabilities as targets. Useful for multiclass probabilities.
/// It stores records as an `Array2` of elements of type `D`, and targets as an `Array1`
/// of elements of type `Pr`
pub type DatasetPr<D> = Dataset<D, Pr>;

/// Records
///
/// The records are input data in the training
pub trait Records: Sized {
    type Elem;

    fn observations(&self) -> usize;
}

/// Targets
pub trait Targets {
    type Elem;

    fn as_slice(&self) -> &[Self::Elem];
}

/// Labels
///
/// Same as targets, but with discrete elements. The labels trait can therefore return the set of
/// labels of the targets
pub trait Labels: Targets
where
    Self::Elem: Label,
{
    fn labels(&self) -> Vec<Self::Elem>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use ndarray_rand::rand::SeedableRng;
    use rand_isaac::Isaac64Rng;

    #[test]
    fn dataset_implements_required_methods() {
        let mut rng = Isaac64Rng::seed_from_u64(42);

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
        let mut rng = Isaac64Rng::seed_from_u64(42);
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
}
