//! Common metrics for regression
//!
//! This module implements common comparison metrices for continuous variables.

use ndarray::prelude::*;
use ndarray::{Data, NdFloat};
use num_traits::FromPrimitive;

/// Regression metrices trait
pub trait Regression<A, D: Data<Elem = A>> {
    /// Maximal error between two continuous variables
    fn max_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    /// Mean error between two continuous variables
    fn mean_absolute_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    /// Mean squared error between two continuous variables
    fn mean_squared_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    /// Mean squared log error between two continuous variables
    fn mean_squared_log_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    /// Median absolute error between two continuous variables
    fn median_absolute_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    /// R squared coefficient, is the proprtion of the variance in the dependent variable that is
    /// predictable from the independent variable
    fn r2(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    /// Same as R-Squared but with biased variance
    fn explained_variance(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
}

impl<A: NdFloat + FromPrimitive, D: Data<Elem = A>> Regression<A, D> for ArrayBase<D, Ix1> {
    fn max_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        (self - compare_to)
            .iter()
            .map(|x| x.abs())
            .fold(A::neg_infinity(), A::max)
    }

    fn mean_absolute_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        (self - compare_to).mapv(|x| x.abs()).mean().unwrap()
    }

    fn mean_squared_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        (self - compare_to).mapv(|x| x * x).mean().unwrap()
    }

    fn mean_squared_log_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        //(self - compare_to).mapv(|x| (x.ln() * x.ln()).mean().unwrap()
        self.mapv(|x| (A::one() + x).ln())
            .mean_squared_error(&compare_to.mapv(|x| (A::one() + x).ln()))
    }

    fn median_absolute_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        let mut abs_error = (self - compare_to).mapv(|x| x.abs()).to_vec();
        abs_error.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let mid = abs_error.len() / 2;
        if abs_error.len() % 2 == 0 {
            (abs_error[mid - 1] + abs_error[mid]) / A::from(2.0).unwrap()
        } else {
            abs_error[mid]
        }
    }

    fn r2(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        let mean = compare_to.mean().unwrap();

        A::one()
            - (self - compare_to).mapv(|x| x * x).sum()
                / (self.mapv(|x| (x - mean) * (x - mean)).sum() + A::from(1e-10).unwrap())
    }

    fn explained_variance(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        let mean = compare_to.mean().unwrap();
        let mean_error = (self - compare_to).mean().unwrap();

        A::one()
            - ((self - compare_to).mapv(|x| x * x).sum() - mean_error)
                / (self.mapv(|x| (x - mean) * (x - mean)).sum() + A::from(1e-10).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::Regression;
    use ndarray::prelude::*;

    #[test]
    fn test_same() {
        let a: Array1<f32> = Array1::ones(100);

        assert_eq!(a.max_error(&a), 0.0);
        assert_eq!(a.mean_absolute_error(&a), 0.0);
        assert_eq!(a.mean_squared_error(&a), 0.0);
        assert_eq!(a.mean_squared_log_error(&a), 0.0);
        assert_eq!(a.median_absolute_error(&a), 0.0);
        assert_eq!(a.r2(&a), 1.0);
        assert_eq!(a.explained_variance(&a), 1.0);
    }

    #[test]
    fn test_max_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.3, 0.2, 0.5, 0.7];

        assert!((a.max_error(&b) - 0.3f32).abs() < 1e-5);
    }

    #[test]
    fn test_median_absolute_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.3, 0.2, 0.5, 0.7];
        // 0.1, 0.2, 0.0, 0.2, 0.3 -> median error is 0.2

        assert!((a.median_absolute_error(&b) - 0.2f32).abs() < 1e-5);
    }

    #[test]
    fn test_mean_squared_error() {
        let a = array![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = array![0.1, 0.2, 0.3, 0.4, 0.5];

        assert!((a.mean_squared_error(&b) - 0.1) < 1e-5);
    }
}
