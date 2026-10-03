use crate::float::Float;
use linfa::dataset::Records;
use ndarray::linalg::Dot;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use sprs::{CsMatBase, SpIndex};
use std::ops::Deref;

/// Feature matrix of a logistic regression problem.
///
/// The solver only needs matrix-vector and matrix-matrix products with the feature matrix and
/// its transpose, so this trait abstracts over dense `ndarray` arrays and sparse `sprs` matrices.
pub trait DesignMatrix<F: Float>: Records<Elem = F> {
    /// Whether every stored value is finite.
    fn all_finite(&self) -> bool;

    /// `X . v` where `v` has `n_features` entries; the result has `n_samples` entries.
    fn dot_vec<S: Data<Elem = F>>(&self, v: &ArrayBase<S, Ix1>) -> Array1<F>;

    /// `X . M` where `M` has shape `(n_features, k)`; the result has shape `(n_samples, k)`.
    fn dot_mat<S: Data<Elem = F>>(&self, m: &ArrayBase<S, Ix2>) -> Array2<F>;

    /// `X^T . v` where `v` has `n_samples` entries; the result has `n_features` entries.
    fn t_dot_vec<S: Data<Elem = F>>(&self, v: &ArrayBase<S, Ix1>) -> Array1<F>;

    /// `X^T . M` where `M` has shape `(n_samples, k)`; the result has shape `(n_features, k)`.
    fn t_dot_mat<S: Data<Elem = F>>(&self, m: &ArrayBase<S, Ix2>) -> Array2<F>;
}

impl<F: Float, D: Data<Elem = F>> DesignMatrix<F> for ArrayBase<D, Ix2> {
    fn all_finite(&self) -> bool {
        self.iter().all(|x| x.is_finite())
    }

    fn dot_vec<S: Data<Elem = F>>(&self, v: &ArrayBase<S, Ix1>) -> Array1<F> {
        self.dot(v)
    }

    fn dot_mat<S: Data<Elem = F>>(&self, m: &ArrayBase<S, Ix2>) -> Array2<F> {
        self.dot(m)
    }

    fn t_dot_vec<S: Data<Elem = F>>(&self, v: &ArrayBase<S, Ix1>) -> Array1<F> {
        self.t().dot(v)
    }

    fn t_dot_mat<S: Data<Elem = F>>(&self, m: &ArrayBase<S, Ix2>) -> Array2<F> {
        self.t().dot(m)
    }
}

/// `sprs` returns column-major results for narrow right-hand sides; normalize to the row-major
/// layout the dense implementation produces so callers see the same layout for both.
fn standard_layout<F: Float>(m: Array2<F>) -> Array2<F> {
    if m.is_standard_layout() {
        m
    } else {
        m.as_standard_layout().into_owned()
    }
}

impl<F, I, Iptr, IpS, IS, DS> DesignMatrix<F> for CsMatBase<F, I, IpS, IS, DS, Iptr>
where
    F: Float,
    I: SpIndex,
    Iptr: SpIndex,
    IpS: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
{
    fn all_finite(&self) -> bool {
        self.data().iter().all(|x| x.is_finite())
    }

    fn dot_vec<S: Data<Elem = F>>(&self, v: &ArrayBase<S, Ix1>) -> Array1<F> {
        // `sprs` reshapes the vector internally, which requires a contiguous layout.
        self.dot(&v.as_standard_layout())
    }

    fn dot_mat<S: Data<Elem = F>>(&self, m: &ArrayBase<S, Ix2>) -> Array2<F> {
        standard_layout(self.dot(m))
    }

    fn t_dot_vec<S: Data<Elem = F>>(&self, v: &ArrayBase<S, Ix1>) -> Array1<F> {
        self.transpose_view().dot(&v.as_standard_layout())
    }

    fn t_dot_mat<S: Data<Elem = F>>(&self, m: &ArrayBase<S, Ix2>) -> Array2<F> {
        standard_layout(self.transpose_view().dot(m))
    }
}
