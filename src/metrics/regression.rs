use ndarray::{Data, NdFloat};
use ndarray::prelude::*;
use num_traits::FromPrimitive;

pub trait Regression<A, D: Data<Elem = A>> {
    fn max_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    fn mean_absolute_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    fn mean_squared_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    fn mean_squared_log_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    fn median_absolute_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    fn r2(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
    fn explained_variance(&self, compare_to: &ArrayBase<D, Ix1>) -> A;
}

impl<A: NdFloat + FromPrimitive, D: Data<Elem = A>> Regression<A, D> for ArrayBase<D, Ix1> {
    fn max_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        (self - compare_to).iter().map(|x| x.abs()).fold(A::neg_infinity(), A::max)
    }

    fn mean_absolute_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        (self - compare_to).mapv(|x| x.abs()).mean().unwrap()
    }

    fn mean_squared_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        (self - compare_to).mapv(|x| x * x).mean().unwrap()
    }

    fn mean_squared_log_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        (self - compare_to).mapv(|x| (x * x).ln()).mean().unwrap()
    }

    fn median_absolute_error(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        let mut abs_error = (self - compare_to).mapv(|x| x.abs()).to_vec();
        abs_error.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let mid = abs_error.len() / 2;
        if abs_error.len() % 2 == 0 {
            (abs_error[mid-1] + abs_error[mid]) / A::from(2.0).unwrap()
        } else {
            abs_error[mid]
        }
    }

    fn r2(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        let mean = compare_to.mean().unwrap();

        A::one() - (self - compare_to).mapv(|x| x*x).sum() / self.mapv(|x| (x - mean) * (x - mean)).sum()
    }

    fn explained_variance(&self, compare_to: &ArrayBase<D, Ix1>) -> A {
        let mean = compare_to.mean().unwrap();
        let mean_error = (self - compare_to).mean().unwrap();

        A::one() - ((self - compare_to).mapv(|x| x*x).sum() - mean_error) / self.mapv(|x| (x - mean) * (x - mean)).sum()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use super::Regression;

    #[test]
    fn test_max_error() {
        let a: Array1<f32> = Array1::ones(100);

        assert_eq!(a.max_error(&a), 0.0);
    }
}

