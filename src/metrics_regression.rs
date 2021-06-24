//! Common metrics for regression
//!
//! This module implements common comparison metrices for continuous variables.

use crate::{
    dataset::{AsTargets, DatasetBase},
    error::{Error, Result},
    Float,
};
use ndarray::prelude::*;
use ndarray::Data;
use std::ops::Sub;

/// Regression metrices trait for single targets.
///
/// It is possible to compute the listed mectrics between:
/// * One-dimensional array - One-dimensional array
/// * One-dimensional array - bi-dimensional array
/// * One-dimensional array - dataset
///
/// In the last two cases, if the second item does not represent a single target,
/// the result will be an error.
///
/// To compare bi-dimensional arrays use [`MultiTargetRegression`](trait.MultiTargetRegression.html)
pub trait SingleTargetRegression<F: Float, T: AsTargets<Elem = F>>: AsTargets<Elem = F> {
    /// Maximal error between two continuous variables
    fn max_error(&self, compare_to: &T) -> Result<F> {
        let max_error = self
            .try_single_target()?
            .sub(&compare_to.try_single_target()?)
            .iter()
            .map(|x| x.abs())
            .fold(F::neg_infinity(), F::max);
        Ok(max_error)
    }
    /// Mean error between two continuous variables
    fn mean_absolute_error(&self, compare_to: &T) -> Result<F> {
        self.try_single_target()?
            .sub(&compare_to.try_single_target()?)
            .mapv(|x| x.abs())
            .mean()
            .ok_or(Error::NotEnoughSamples)
    }

    /// Mean squared error between two continuous variables
    fn mean_squared_error(&self, compare_to: &T) -> Result<F> {
        self.try_single_target()?
            .sub(&compare_to.try_single_target()?)
            .mapv(|x| x * x)
            .mean()
            .ok_or(Error::NotEnoughSamples)
    }

    /// Mean squared log error between two continuous variables
    fn mean_squared_log_error(&self, compare_to: &T) -> Result<F> {
        self.try_single_target()?
            .mapv(|x| (F::one() + x).ln())
            .mean_squared_error(
                &compare_to
                    .try_single_target()?
                    .mapv(|x| (F::one() + x).ln()),
            )
    }

    /// Median absolute error between two continuous variables
    fn median_absolute_error(&self, compare_to: &T) -> Result<F> {
        let mut abs_error = self
            .try_single_target()?
            .sub(&compare_to.try_single_target()?)
            .mapv(|x| x.abs())
            .to_vec();
        abs_error.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = abs_error.len() / 2;
        if abs_error.len() % 2 == 0 {
            Ok((abs_error[mid - 1] + abs_error[mid]) / F::cast(2.0))
        } else {
            Ok(abs_error[mid])
        }
    }

    /// R squared coefficient, is the proportion of the variance in the dependent variable that is
    /// predictable from the independent variable
    // r2 = 1 - sum((pred_i - y_i)^2)/sum((mean_y - y_i)^2)
    // if the mean is of `compare_to`, then the denominator
    // should compare `compare_to` and the mean, and not self and the mean
    fn r2(&self, compare_to: &T) -> Result<F> {
        let single_target_compare_to = compare_to.try_single_target()?;
        let mean = single_target_compare_to
            .mean()
            .ok_or(Error::NotEnoughSamples)?;

        Ok(F::one()
            - self
                .try_single_target()?
                .sub(&single_target_compare_to)
                .mapv(|x| x * x)
                .sum()
                / (single_target_compare_to
                    .mapv(|x| (x - mean) * (x - mean))
                    .sum()
                    + F::cast(1e-10)))
    }

    /// Same as R-Squared but with biased variance
    fn explained_variance(&self, compare_to: &T) -> Result<F> {
        let single_target_compare_to = compare_to.try_single_target()?;
        let diff = self.try_single_target()?.sub(&single_target_compare_to);

        let mean = single_target_compare_to
            .mean()
            .ok_or(Error::NotEnoughSamples)?;
        let mean_error = diff.mean().ok_or(Error::NotEnoughSamples)?;

        Ok(F::one()
            - (diff.mapv(|x| x * x).sum() - mean_error)
                / (single_target_compare_to
                    .mapv(|x| (x - mean) * (x - mean))
                    .sum()
                    + F::cast(1e-10)))
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsTargets<Elem = F>> SingleTargetRegression<F, T>
    for ArrayBase<D, Ix1>
{
}

/// Regression metrices trait for multiple targets.
///
/// It is possible to compute the listed mectrics between:
/// * bi-dimensional array - bi-dimensional array
/// * bi-dimensional array - dataset
/// * dataset - dataset
/// * dataset - one-dimensional array
/// * dataset - bi-dimensional array
///
/// The shape of the compared targets must match.
///
/// To compare single-dimensional arrays use [`SingleTargetRegression`](trait.SingleTargetRegression.html)
pub trait MultiTargetRegression<F: Float, T: AsTargets<Elem = F>>: AsTargets<Elem = F> {
    /// Maximal error between two continuous variables
    fn max_error(&self, other: &T) -> Result<Array1<F>> {
        self.as_multi_targets()
            .axis_iter(Axis(1))
            .zip(other.as_multi_targets().axis_iter(Axis(1)))
            .map(|(a, b)| a.max_error(&b))
            .collect()
    }
    /// Mean error between two continuous variables
    fn mean_absolute_error(&self, other: &T) -> Result<Array1<F>> {
        self.as_multi_targets()
            .axis_iter(Axis(1))
            .zip(other.as_multi_targets().axis_iter(Axis(1)))
            .map(|(a, b)| a.mean_absolute_error(&b))
            .collect()
    }

    /// Mean squared error between two continuous variables
    fn mean_squared_error(&self, other: &T) -> Result<Array1<F>> {
        self.as_multi_targets()
            .axis_iter(Axis(1))
            .zip(other.as_multi_targets().axis_iter(Axis(1)))
            .map(|(a, b)| a.mean_squared_error(&b))
            .collect()
    }

    /// Mean squared log error between two continuous variables
    fn mean_squared_log_error(&self, other: &T) -> Result<Array1<F>> {
        self.as_multi_targets()
            .axis_iter(Axis(1))
            .zip(other.as_multi_targets().axis_iter(Axis(1)))
            .map(|(a, b)| a.mean_squared_log_error(&b))
            .collect()
    }

    /// Median absolute error between two continuous variables
    fn median_absolute_error(&self, other: &T) -> Result<Array1<F>> {
        self.as_multi_targets()
            .axis_iter(Axis(1))
            .zip(other.as_multi_targets().axis_iter(Axis(1)))
            .map(|(a, b)| a.median_absolute_error(&b))
            .collect()
    }

    /// R squared coefficient, is the proportion of the variance in the dependent variable that is
    /// predictable from the independent variable
    fn r2(&self, other: &T) -> Result<Array1<F>> {
        self.as_multi_targets()
            .axis_iter(Axis(1))
            .zip(other.as_multi_targets().axis_iter(Axis(1)))
            .map(|(a, b)| a.r2(&b))
            .collect()
    }

    /// Same as R-Squared but with biased variance
    fn explained_variance(&self, other: &T) -> Result<Array1<F>> {
        self.as_multi_targets()
            .axis_iter(Axis(1))
            .zip(other.as_multi_targets().axis_iter(Axis(1)))
            .map(|(a, b)| a.explained_variance(&b))
            .collect()
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsTargets<Elem = F>> MultiTargetRegression<F, T>
    for ArrayBase<D, Ix2>
{
}

impl<F: Float, T: AsTargets<Elem = F>, T2: AsTargets<Elem = F>, D: Data<Elem = F>>
    MultiTargetRegression<F, T2> for DatasetBase<ArrayBase<D, Ix2>, T>
{
}

#[cfg(test)]
mod tests {
    use super::{MultiTargetRegression, SingleTargetRegression};
    use crate::dataset::DatasetBase;
    use approx::assert_abs_diff_eq;
    use ndarray::prelude::*;

    #[test]
    fn test_same() {
        let a: Array1<f32> = Array1::ones(100);

        assert_abs_diff_eq!(a.max_error(&a).unwrap(), 0.0f32);
        assert_abs_diff_eq!(a.mean_absolute_error(&a).unwrap(), 0.0f32);
        assert_abs_diff_eq!(a.mean_squared_error(&a).unwrap(), 0.0f32);
        assert_abs_diff_eq!(a.mean_squared_log_error(&a).unwrap(), 0.0f32);
        assert_abs_diff_eq!(a.median_absolute_error(&a).unwrap(), 0.0f32);
        assert_abs_diff_eq!(a.r2(&a).unwrap(), 1.0f32);
        assert_abs_diff_eq!(a.explained_variance(&a).unwrap(), 1.0f32);
    }

    #[test]
    fn test_max_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.3, 0.2, 0.5, 0.7];

        assert_abs_diff_eq!(a.max_error(&b).unwrap(), 0.3f32, epsilon = 1e-5);
    }

    #[test]
    fn test_median_absolute_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.3, 0.2, 0.5, 0.7];
        // 0.1, 0.2, 0.0, 0.2, 0.3 -> median error is 0.2

        assert_abs_diff_eq!(a.median_absolute_error(&b).unwrap(), 0.2f32, epsilon = 1e-5);
    }

    #[test]
    fn test_mean_squared_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.2, 0.3, 0.4, 0.5];

        assert_abs_diff_eq!(a.mean_squared_error(&b).unwrap(), 0.01, epsilon = 1e-5);
    }

    #[test]
    fn test_max_error_for_single_targets() {
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]];
        let targets = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction: Array1<f64> = array![0.1, 0.3, 0.2, 0.5, 0.7];
        let abs_err_from_arr1 = prediction.max_error(st_dataset.targets()).unwrap();
        let prediction: DatasetBase<_, _> = (records.view(), prediction.view()).into();
        let abs_err_from_ds = prediction.max_error(&st_dataset.targets()).unwrap();
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
        let abs_err_from_arr1 = prediction.mean_absolute_error(&st_dataset).unwrap();
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction
            .mean_absolute_error(st_dataset.targets())
            .unwrap();
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
        let abs_err_from_arr1 = prediction.mean_squared_error(st_dataset.targets()).unwrap();
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.mean_squared_error(st_dataset.targets()).unwrap();
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
        let abs_err_from_arr1 = prediction
            .mean_squared_log_error(st_dataset.targets())
            .unwrap();
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction
            .mean_squared_log_error(st_dataset.targets())
            .unwrap();
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.019_033, epsilon = 1e-5);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);
    }

    #[test]
    fn test_median_absolute_error_for_single_targets() {
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3],];
        let targets = array![0.0, 0.1, 0.2, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.7];
        // even length absolute errors
        let abs_err_from_arr1 = prediction
            .median_absolute_error(st_dataset.targets())
            .unwrap();
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction
            .median_absolute_error(st_dataset.targets())
            .unwrap();
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.15, epsilon = 1e-5);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);

        // odd length absolute errors
        let records = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]];
        let targets = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let st_dataset: DatasetBase<_, _> = (records.view(), targets).into();
        let prediction = array![0.1, 0.3, 0.2, 0.51, 0.7];
        let abs_err_from_arr1 = prediction.median_absolute_error(&st_dataset).unwrap();
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.median_absolute_error(&st_dataset).unwrap();
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
        let abs_err_from_arr1 = prediction.r2(st_dataset.targets()).unwrap();
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.r2(st_dataset.targets()).unwrap();
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
        let abs_err_from_arr1 = prediction.explained_variance(st_dataset.targets()).unwrap();
        let prediction: DatasetBase<_, _> = (records.view(), prediction).into();
        let abs_err_from_ds = prediction.explained_variance(&st_dataset).unwrap();
        assert_eq!(abs_err_from_ds.dim(), 1);
        assert_abs_diff_eq!(abs_err_from_arr1, 0.8, epsilon = 1e-5);
        assert_abs_diff_eq!(abs_err_from_arr1, abs_err_from_ds[0]);
    }
}
