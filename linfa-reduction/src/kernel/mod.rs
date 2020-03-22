use ndarray::prelude::*;
use ndarray::Data;
use ndarray::NdFloat;
use num_traits::{FromPrimitive, Num, NumCast};
use sprs::CsMat;

pub enum Axis {
    Column,
    Row
}

pub trait SimilarityKernel {}

pub trait Kernel<A> {
    fn apply_gram<S: Data<Elem = A>>(&self, rhs: ArrayBase<S, Ix2>) -> Array2<A>;
    fn mean(&self, axis: Axis) -> Array1<A>;
    fn n_features(&self) -> usize;
}

impl<S: Data<Elem = A>, A: NdFloat + 'static + FromPrimitive> Kernel<A> for ArrayBase<S, Ix2> {
    fn apply_gram<T: Data<Elem = A>>(&self, rhs: ArrayBase<T, Ix2>) -> Array2<A> {
        self.dot(&rhs)
    }

    fn mean(&self, axis: Axis) -> Array1<A> {
        match axis {
            Axis::Row => self.mean_axis(ndarray::Axis(0)).unwrap(),
            Axis::Column => self.mean_axis(ndarray::Axis(1)).unwrap()
        }
    }

    fn n_features(&self) -> usize {
        self.ncols()
    }
}

impl<A: NdFloat + 'static + FromPrimitive + Num + Default> Kernel<A> for CsMat<A> {
    fn apply_gram<S: Data<Elem = A>>(&self, rhs: ArrayBase<S, Ix2>) -> Array2<A> {
        self * &rhs
    }

    fn mean(&self, axis: Axis) -> Array1<A> {
        let mut mean: Array1<A> = match axis {
            Axis::Column => Array1::zeros(self.cols()),
            Axis::Row => Array1::zeros(self.rows())
        };

        for (val, (row, col)) in self.iter() {
            match axis {
                Axis::Column => mean[col] += *val,
                Axis::Row => mean[row] += *val
            }
        }

        let cols: A = NumCast::from(self.cols()).unwrap();
        let rows: A = NumCast::from(self.rows()).unwrap();

        match axis {
            Axis::Column => mean / rows,
            Axis::Row => mean / cols
        }
    }

    fn n_features(&self) -> usize {
        self.cols()
    }
}

#[cfg(test)]
mod tests {
    use super::{Kernel, Axis};
    use sprs::CsMatBase;
    use ndarray::{Array1, Array2};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    #[test]
    fn test_sprs() {
        let a: Array2<f64> = Array2::random((10, 5), Uniform::new(0., 1.));
        let a = CsMatBase::csr_from_dense(a.view(), 1e-5);

        assert_eq!(a.mean(Axis::Column).shape(), &[5]);
        assert_eq!(a.mean(Axis::Row).shape(), &[10]);
        assert_eq!(a.n_features(), 5);

        assert_eq!(a.apply_gram(Array2::eye(5)), a.to_dense());
    }

    #[test]
    fn test_dense() {
        let id: Array2<f64> = Array2::eye(10);

        assert_eq!(Kernel::mean(&id, Axis::Column), Array1::ones(10) * 0.1);
        assert_eq!(Kernel::mean(&id, Axis::Row), Array1::ones(10) * 0.1);

        assert_eq!(id.n_features(), 10);
    }
}
