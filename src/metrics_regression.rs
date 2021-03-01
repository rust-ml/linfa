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
    T: IntoNdProducer<Item = &'a A, Dim = Ix1, Output = ArrayView1<'a, A>>,
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
    /// R squared coefficient, is the proprtion of the variance in the dependent variable that is
    /// predictable from the independent variable
    fn r2(&self, compare_to: T) -> A;
    /// Same as R-Squared but with biased variance
    fn explained_variance(&self, compare_to: T) -> A;
}

impl<
        'a,
        A: 'a + NdFloat + FromPrimitive,
        D: Data<Elem = A>,
        T: IntoNdProducer<Item = &'a A, Dim = Ix1, Output = ArrayView1<'a, A>>,
    > Regression<'a, A, T> for ArrayBase<D, Ix1>
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

    fn r2(&self, compare_to: T) -> A {
        let compare_to = compare_to.into_producer();

        let mean = compare_to.mean().unwrap();

        A::one()
            - self.sub(&compare_to).mapv(|x| x * x).sum()
                / (self.mapv(|x| (x - mean) * (x - mean)).sum() + A::from(1e-10).unwrap())
    }

    fn explained_variance(&self, compare_to: T) -> A {
        let compare_to = compare_to.into_producer();
        let diff = self.sub(&compare_to);

        let mean = compare_to.mean().unwrap();
        let mean_error = diff.mean().unwrap();

        A::one()
            - (diff.mapv(|x| x * x).sum() - mean_error)
                / (self.mapv(|x| (x - mean) * (x - mean)).sum() + A::from(1e-10).unwrap())
    }
}

impl<F: Float, R: Records, T: AsTargets<Elem = F>> DatasetBase<R, T> {
    /// Maximal error between two continuous variables
    pub fn max_error(&self, compare_to: T) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.max_error(b))
            .collect()
    }

    /// Mean error between two continuous variables
    pub fn mean_absolute_error(&self, compare_to: T) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.mean_absolute_error(b))
            .collect()
    }

    /// Mean squared error between two continuous variables
    pub fn mean_squared_error(&self, compare_to: T) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.mean_squared_error(b))
            .collect()
    }

    /// Mean squared log error between two continuous variables
    pub fn mean_squared_log_error(&self, compare_to: T) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.mean_squared_log_error(b))
            .collect()
    }

    /// Median absolute error between two continuous variables
    pub fn median_absolute_error(&self, compare_to: T) -> Array1<F> {
        let t1 = self.as_multi_targets();
        let t2 = compare_to.as_multi_targets();

        t1.gencolumns()
            .into_iter()
            .zip(t2.gencolumns().into_iter())
            .map(|(a, b)| a.median_absolute_error(b))
            .collect()
    }

    /// R squared coefficient, is the proprtion of the variance in the dependent variable that is
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
    pub fn explained_variance(&self, compare_to: T) -> Array1<F> {
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
    use approx::abs_diff_eq;
    use ndarray::prelude::*;

    #[test]
    fn test_same() {
        let a: Array1<f32> = Array1::ones(100);

        abs_diff_eq!(a.max_error(&a), 0.0f32);
        abs_diff_eq!(a.mean_absolute_error(&a), 0.0f32);
        abs_diff_eq!(a.mean_squared_error(&a), 0.0f32);
        abs_diff_eq!(a.mean_squared_log_error(&a), 0.0f32);
        abs_diff_eq!(a.median_absolute_error(&a), 0.0f32);
        abs_diff_eq!(a.r2(&a), 1.0f32);
        abs_diff_eq!(a.explained_variance(&a), 1.0f32);
    }

    #[test]
    fn test_max_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.3, 0.2, 0.5, 0.7];

        abs_diff_eq!(a.max_error(&b), 0.3f32, epsilon = 1e-5);
    }

    #[test]
    fn test_median_absolute_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.3, 0.2, 0.5, 0.7];
        // 0.1, 0.2, 0.0, 0.2, 0.3 -> median error is 0.2

        abs_diff_eq!(a.median_absolute_error(&b), 0.2f32, epsilon = 1e-5);
    }

    #[test]
    fn test_mean_squared_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.2, 0.3, 0.4, 0.5];

        abs_diff_eq!(a.mean_squared_error(&b), 0.1, epsilon = 1e-5);
    }
}
