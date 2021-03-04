//! Common metrics for regression
//!
//! This module implements common comparison metrices for continuous variables.

use ndarray::prelude::*;
use ndarray::IntoNdProducer;
use ndarray::{Data, NdFloat};
use num_traits::FromPrimitive;
use std::ops::Sub;

use crate::{
    dataset::{AsTargets, DatasetBase, Records},
    Float,
};

/// Regression metrices trait
pub trait Regression<
    'a,
    A: 'a,
    D: ndarray::Dimension,
    T: IntoNdProducer<Item = &'a A, Dim = D, Output = ArrayView<'a, A, D>>,
>
{
    /// Maximal error between two continuous variables
    fn max_error(&self, compare_to: T) -> A;
    /// Mean error between two continuous variables
    fn mean_absolute_error(&self, compare_to: T) -> A;
    /// Mean squared error between two continuous variables
    fn mean_squared_error(&self, compare_to: T) -> A;
    /// Mean squared log error between two continuous variables
    fn mean_squared_log_error(&self, compare_to: T) -> A;
    /// Median absolute error between two continuous variables
    fn median_absolute_error(&self, compare_to: T) -> A;
    /// R squared coefficient, is the proportion of the variance in the dependent variable that is
    /// predictable from the independent variable.
    ///
    /// To evaluate the accuracy of a prediction, use
    /// ```ignore
    /// prediction.r2(ground_truth)
    /// ```
    fn r2(&self, compare_to: T) -> A;
    /// Same as R-Squared but with biased variance
    fn explained_variance(&self, compare_to: T) -> A;
}

impl<
        'a,
        A: 'a + NdFloat + FromPrimitive,
        D: Data<Elem = A>,
        T: IntoNdProducer<Item = &'a A, Dim = Ix1, Output = ArrayView<'a, A, Ix1>>,
    > Regression<'a, A, Ix1, T> for ArrayBase<D, Ix1>
{
    fn max_error(&self, compare_to: T) -> A {
        let compare_to: ArrayView1<'a, A> = compare_to.into_producer();

        self.sub(&compare_to)
            .iter()
            .map(|x| x.abs())
            .fold(A::neg_infinity(), A::max)
    }

    fn mean_absolute_error(&self, compare_to: T) -> A {
        let compare_to = compare_to.into_producer();

        self.sub(&compare_to).mapv(|x| x.abs()).mean().unwrap()
    }

    fn mean_squared_error(&self, compare_to: T) -> A {
        let compare_to = compare_to.into_producer();

        self.sub(&compare_to).mapv(|x| x * x).mean().unwrap()
    }

    fn mean_squared_log_error(&self, compare_to: T) -> A {
        let compare_to = compare_to.into_producer();

        self.mapv(|x| (A::one() + x).ln())
            .mean_squared_error(&compare_to.mapv(|x| (A::one() + x).ln()))
    }

    fn median_absolute_error(&self, compare_to: T) -> A {
        let compare_to = compare_to.into_producer();

        let mut abs_error = self.sub(&compare_to).mapv(|x| x.abs()).to_vec();
        abs_error.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let mid = abs_error.len() / 2;
        if abs_error.len() % 2 == 0 {
            (abs_error[mid - 1] + abs_error[mid]) / A::from(2.0).unwrap()
        } else {
            abs_error[mid]
        }
    }

    // r2 = 1 - sum((pred_i - y_i)^2)/sum((mean_y - y_i)^2)
    // if the mean is of `compare_to`, then the denominator
    // should compare `compare_to` and the mean, and not self and the mean
    fn r2(&self, compare_to: T) -> A {
        let compare_to = compare_to.into_producer();

        let mean = compare_to.mean().unwrap();

        A::one()
            - self.sub(&compare_to).mapv(|x| x * x).sum()
                / (compare_to.mapv(|x| (x - mean) * (x - mean)).sum() + A::from(1e-10).unwrap())
    }

    fn explained_variance(&self, compare_to: T) -> A {
        let compare_to = compare_to.into_producer();
        let diff = self.sub(&compare_to);

        let mean = compare_to.mean().unwrap();
        let mean_error = diff.mean().unwrap();

        A::one()
            - (diff.mapv(|x| x * x).sum() - mean_error)
                / (compare_to.mapv(|x| (x - mean) * (x - mean)).sum() + A::from(1e-10).unwrap())
    }
}

impl<
        'a,
        A: 'a + NdFloat + FromPrimitive,
        D: Data<Elem = A>,
        T: IntoNdProducer<Item = &'a A, Dim = Ix2, Output = ArrayView<'a, A, Ix2>>,
    > Regression<'a, A, Ix2, T> for ArrayBase<D, Ix1>
{
    fn max_error(&self, compare_to: T) -> A {
        let compare_to: ArrayView2<'a, A> = compare_to.into_producer();
        if compare_to.len_of(Axis(1)) > 1 {
            panic!("Expected single targets array");
        }

        let compare_to = compare_to.index_axis_move(Axis(1), 0);

        self.max_error(compare_to)
    }

    fn mean_absolute_error(&self, compare_to: T) -> A {
        let compare_to: ArrayView2<'a, A> = compare_to.into_producer();
        if compare_to.len_of(Axis(1)) > 1 {
            panic!("Expected single targets array");
        }

        let compare_to = compare_to.index_axis_move(Axis(1), 0);

        self.mean_absolute_error(compare_to)
    }

    fn mean_squared_error(&self, compare_to: T) -> A {
        let compare_to: ArrayView2<'a, A> = compare_to.into_producer();
        if compare_to.len_of(Axis(1)) > 1 {
            panic!("Expected single targets array");
        }

        let compare_to = compare_to.index_axis_move(Axis(1), 0);

        self.mean_squared_error(compare_to)
    }

    fn mean_squared_log_error(&self, compare_to: T) -> A {
        let compare_to: ArrayView2<'a, A> = compare_to.into_producer();
        if compare_to.len_of(Axis(1)) > 1 {
            panic!("Expected single targets array");
        }

        let compare_to = compare_to.index_axis_move(Axis(1), 0);

        self.mean_squared_log_error(compare_to)
    }

    fn median_absolute_error(&self, compare_to: T) -> A {
        let compare_to: ArrayView2<'a, A> = compare_to.into_producer();
        if compare_to.len_of(Axis(1)) > 1 {
            panic!("Expected single targets array");
        }

        let compare_to = compare_to.index_axis_move(Axis(1), 0);

        self.median_absolute_error(compare_to)
    }

    fn r2(&self, compare_to: T) -> A {
        let compare_to: ArrayView2<'a, A> = compare_to.into_producer();
        if compare_to.len_of(Axis(1)) > 1 {
            panic!("Expected single targets array");
        }

        let compare_to = compare_to.index_axis_move(Axis(1), 0);

        self.r2(compare_to)
    }

    fn explained_variance(&self, compare_to: T) -> A {
        let compare_to: ArrayView2<'a, A> = compare_to.into_producer();
        if compare_to.len_of(Axis(1)) > 1 {
            panic!("Expected single targets array");
        }

        let compare_to = compare_to.index_axis_move(Axis(1), 0);

        self.explained_variance(compare_to)
    }
}

impl<F: Float, R: Records, T: AsTargets<Elem = F>> DatasetBase<R, T> {
    /// Maximal error between two continuous variables
    pub fn max_error<T2: AsTargets<Elem = F>>(&self, compare_to: T2) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.max_error(b))
            .collect()
    }

    /// Mean error between two continuous variables
    pub fn mean_absolute_error<T2: AsTargets<Elem = F>>(&self, compare_to: T2) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.mean_absolute_error(b))
            .collect()
    }

    /// Mean squared error between two continuous variables
    pub fn mean_squared_error<T2: AsTargets<Elem = F>>(&self, compare_to: T2) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.mean_squared_error(b))
            .collect()
    }

    /// Mean squared log error between two continuous variables
    pub fn mean_squared_log_error<T2: AsTargets<Elem = F>>(&self, compare_to: T2) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.mean_squared_log_error(b))
            .collect()
    }

    /// Median absolute error between two continuous variables
    pub fn median_absolute_error<T2: AsTargets<Elem = F>>(&self, compare_to: T2) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.median_absolute_error(b))
            .collect()
    }

    /// R squared coefficient, is the proportion of the variance in the dependent variable that is
    /// predictable from the independent variable
    pub fn r2<T2: AsTargets<Elem = F>>(&self, compare_to: T2) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.r2(b))
            .collect()
    }

    /// Same as R-Squared but with biased variance
    pub fn explained_variance<T2: AsTargets<Elem = F>>(&self, compare_to: T2) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.explained_variance(b))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Regression;
    use crate::dataset::DatasetBase;
    use approx::assert_abs_diff_eq;
    use ndarray::prelude::*;

    #[test]
    fn test_same() {
        let a: Array1<f32> = Array1::ones(100);

        assert_abs_diff_eq!(a.max_error(&a), 0.0f32);
        assert_abs_diff_eq!(a.mean_absolute_error(&a), 0.0f32);
        assert_abs_diff_eq!(a.mean_squared_error(&a), 0.0f32);
        assert_abs_diff_eq!(a.mean_squared_log_error(&a), 0.0f32);
        assert_abs_diff_eq!(a.median_absolute_error(&a), 0.0f32);
        assert_abs_diff_eq!(a.r2(&a), 1.0f32);
        assert_abs_diff_eq!(a.explained_variance(&a), 1.0f32);
    }

    #[test]
    fn test_max_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.3, 0.2, 0.5, 0.7];

        assert_abs_diff_eq!(a.max_error(&b), 0.3f32, epsilon = 1e-5);
    }

    #[test]
    fn test_median_absolute_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.3, 0.2, 0.5, 0.7];
        // 0.1, 0.2, 0.0, 0.2, 0.3 -> median error is 0.2

        assert_abs_diff_eq!(a.median_absolute_error(&b), 0.2f32, epsilon = 1e-5);
    }

    #[test]
    fn test_mean_squared_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.2, 0.3, 0.4, 0.5];

        assert_abs_diff_eq!(a.mean_squared_error(&b), 0.01, epsilon = 1e-5);
    }

    #[test]
    fn test_max_error_for_single_targets() {
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]];
        let targets = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.5, 0.7];
        let abs_err_from_arr1 = prediction.max_error(st_dataset.targets());
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.max_error(st_dataset.targets());
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.3);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);
    }

    #[test]
    fn test_mean_absolute_error_for_single_targets() {
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]];
        let targets = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.5, 0.7];
        let abs_err_from_arr1 = prediction.mean_absolute_error(st_dataset.targets());
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.mean_absolute_error(st_dataset.targets());
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.16);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);
    }

    #[test]
    fn test_mean_squared_error_for_single_targets() {
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]];
        let targets = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.5, 0.7];
        let abs_err_from_arr1 = prediction.mean_squared_error(st_dataset.targets());
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.mean_squared_error(st_dataset.targets());
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.036);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);
    }

    #[test]
    fn test_mean_squared_log_error_for_single_targets() {
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]];
        let targets = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.5, 0.7];
        let abs_err_from_arr1 = prediction.mean_squared_log_error(st_dataset.targets());
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.mean_squared_log_error(st_dataset.targets());
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.019033, epsilon = 1e-5);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);
    }

    #[test]
    fn test_median_absolute_error_for_single_targets() {
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3],];
        let targets = array![0.0, 0.1, 0.2, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.7];
        // even length absolute errors
        let abs_err_from_arr1 = prediction.median_absolute_error(st_dataset.targets());
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.median_absolute_error(st_dataset.targets());
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.15, epsilon = 1e-5);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);

        // odd length absolute errors
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]];
        let targets = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.51, 0.7];
        let abs_err_from_arr1 = prediction.median_absolute_error(st_dataset.targets());
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.median_absolute_error(st_dataset.targets());
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.2, epsilon = 1e-5);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);
    }

    #[test]
    fn test_r2_for_single_targets() {
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]];
        let targets = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.5, 0.7];
        let abs_err_from_arr1 = prediction.r2(st_dataset.targets());
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.r2(st_dataset.targets());
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, -0.8, epsilon = 1e-5);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);
    }

    #[test]
    fn test_explained_variance_for_single_targets() {
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]];
        let targets = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.5, 0.7];
        let abs_err_from_arr1 = prediction.explained_variance(st_dataset.targets());
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.explained_variance(st_dataset.targets());
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.8, epsilon = 1e-5);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);
    }
}
