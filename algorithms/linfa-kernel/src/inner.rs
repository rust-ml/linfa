use linfa::Float;
use ndarray::prelude::*;
use ndarray::Data;
use sprs::{CsMat, CsMatView};
use std::ops::Mul;

/// Specifies the methods an inner matrix of a kernel must
/// be able to provide
pub trait Inner {
    type Elem: Float;

    fn dot(&self, rhs: &ArrayView2<Self::Elem>) -> Array2<Self::Elem>;
    fn sum(&self) -> Array1<Self::Elem>;
    fn size(&self) -> usize;
    fn column(&self, i: usize) -> Vec<Self::Elem>;
    fn to_upper_triangle(&self) -> Vec<Self::Elem>;
    fn is_dense(&self) -> bool;
    fn diagonal(&self) -> Array1<Self::Elem>;
}

/// Allows a kernel to have either a dense or a sparse inner
/// matrix in a way that is transparent to the user
pub enum KernelInner<K1: Inner, K2: Inner> {
    Dense(K1),
    Sparse(K2),
}

impl<F: Float, D: Data<Elem = F>> Inner for ArrayBase<D, Ix2> {
    type Elem = F;

    fn dot(&self, rhs: &ArrayView2<F>) -> Array2<F> {
        self.dot(rhs)
    }
    fn sum(&self) -> Array1<F> {
        self.sum_axis(Axis(1))
    }
    fn size(&self) -> usize {
        self.ncols()
    }
    fn column(&self, i: usize) -> Vec<F> {
        self.column(i).to_vec()
    }
    fn to_upper_triangle(&self) -> Vec<F> {
        self.indexed_iter()
            .filter(|((row, col), _)| col > row)
            .map(|(_, val)| *val)
            .collect()
    }

    fn diagonal(&self) -> Array1<F> {
        self.diag().to_owned()
    }

    fn is_dense(&self) -> bool {
        true
    }
}

impl<F: Float> Inner for CsMat<F> {
    type Elem = F;

    fn dot(&self, rhs: &ArrayView2<F>) -> Array2<F> {
        self.mul(rhs)
    }
    fn sum(&self) -> Array1<F> {
        let mut sum = Array1::zeros(self.cols());
        for (val, i) in self.iter() {
            let (_, col) = i;
            sum[col] += *val;
        }

        sum
    }
    fn size(&self) -> usize {
        self.cols()
    }
    fn column(&self, i: usize) -> Vec<F> {
        (0..self.size())
            .map(|j| *self.get(j, i).unwrap_or(&F::neg_zero()))
            .collect::<Vec<_>>()
    }
    fn to_upper_triangle(&self) -> Vec<F> {
        let mat = self.to_dense();
        mat.indexed_iter()
            .filter(|((row, col), _)| col > row)
            .map(|(_, val)| *val)
            .collect()
    }

    fn diagonal(&self) -> Array1<F> {
        let diag_sprs = self.diag();
        let mut diag = Array1::zeros(diag_sprs.dim());
        for (sparse_i, sparse_elem) in diag_sprs.iter() {
            diag[sparse_i] = *sparse_elem;
        }
        diag
    }

    fn is_dense(&self) -> bool {
        false
    }
}

impl<'a, F: Float> Inner for CsMatView<'a, F> {
    type Elem = F;

    fn dot(&self, rhs: &ArrayView2<F>) -> Array2<F> {
        self.mul(rhs)
    }
    fn sum(&self) -> Array1<F> {
        let mut sum = Array1::zeros(self.cols());
        for (val, i) in self.iter() {
            let (_, col) = i;
            sum[col] += *val;
        }

        sum
    }
    fn size(&self) -> usize {
        self.cols()
    }
    fn column(&self, i: usize) -> Vec<F> {
        (0..self.size())
            .map(|j| *self.get(j, i).unwrap_or(&F::neg_zero()))
            .collect::<Vec<_>>()
    }
    fn to_upper_triangle(&self) -> Vec<F> {
        let mat = self.to_dense();
        mat.indexed_iter()
            .filter(|((row, col), _)| col > row)
            .map(|(_, val)| *val)
            .collect()
    }
    fn diagonal(&self) -> Array1<F> {
        let diag_sprs = self.diag();
        let mut diag = Array1::zeros(diag_sprs.dim());
        for (sparse_i, sparse_elem) in diag_sprs.iter() {
            diag[sparse_i] = *sparse_elem;
        }
        diag
    }
    fn is_dense(&self) -> bool {
        false
    }
}
