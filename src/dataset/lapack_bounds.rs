use crate::Float;
use ndarray::{ArrayBase, Data, Dimension, OwnedRepr, ViewRepr};

/// Add the Lapack bound to the floating point of a dataset
///
/// This helper trait is introduced to avoid leaking `Lapack + Scalar` bounds to the outside which
/// causes ambiguities when calling functions like `abs` for `num_traits::Float` and
/// `Cauchy::Scalar`. We are only using real values here, but the LAPACK routines
/// require that `Cauchy::Scalar` is implemented.
pub trait WithLapack<D: Data + WithLapackData, I: Dimension> {
    fn with_lapack(self) -> ArrayBase<D::D, I>;
}

impl<F: Float, D, I> WithLapack<D, I> for ArrayBase<D, I>
where
    D: Data<Elem = F> + WithLapackData,
    I: Dimension,
{
    fn with_lapack(self) -> ArrayBase<D::D, I> {
        D::with_lapack(self)
    }
}

/// Remove the Lapack bound to the floating point of a dataset
///
/// This helper trait is introduced to avoid leaking `Lapack + Scalar` bounds to the outside which
/// causes ambiguities when calling functions like `abs` for `num_traits::Float` and
/// `Cauchy::Scalar`. We are only using real values here, but the LAPACK routines
/// require that `Cauchy::Scalar` is implemented.
pub trait WithoutLapack<F: Float, D: Data + WithoutLapackData<F>, I: Dimension> {
    fn without_lapack(self) -> ArrayBase<D::D, I>;
}

impl<F: Float, D, I> WithoutLapack<F, D, I> for ArrayBase<D, I>
where
    D: Data<Elem = F::Lapack> + WithoutLapackData<F>,
    I: Dimension,
{
    fn without_lapack(self) -> ArrayBase<D::D, I> {
        D::without_lapack(self)
    }
}

unsafe fn transmute<A, B>(a: A) -> B {
    let b = std::ptr::read(&a as *const A as *const B);
    std::mem::forget(a);

    b
}

pub trait WithLapackData
where
    Self: Data,
{
    type D: Data;

    /// Add trait bound `Lapack` and `Scalar` to NdArray's floating point
    ///
    /// This is safe, because only implemented for D == Self
    fn with_lapack<I>(x: ArrayBase<Self, I>) -> ArrayBase<Self::D, I>
    where
        I: Dimension,
    {
        unsafe { transmute(x) }
    }
}

impl<F: Float> WithLapackData for OwnedRepr<F> {
    type D = OwnedRepr<F::Lapack>;
}

impl<'a, F: Float> WithLapackData for ViewRepr<&'a F> {
    type D = ViewRepr<&'a F::Lapack>;
}

impl<'a, F: Float> WithLapackData for ViewRepr<&'a mut F> {
    type D = ViewRepr<&'a mut F::Lapack>;
}

pub trait WithoutLapackData<F: Float>
where
    Self: Data,
{
    type D: Data<Elem = F>;

    /// Add trait bound `Lapack` and `Scalar` to NdArray's floating point
    ///
    /// This is safe, because only implemented for D == Self
    fn without_lapack<I>(x: ArrayBase<Self, I>) -> ArrayBase<Self::D, I>
    where
        I: Dimension,
    {
        unsafe { transmute(x) }
    }
}

impl<F: Float> WithoutLapackData<F> for OwnedRepr<F::Lapack> {
    type D = OwnedRepr<F>;
}

impl<'a, F: Float> WithoutLapackData<F> for ViewRepr<&'a F::Lapack> {
    type D = ViewRepr<&'a F>;
}

impl<'a, F: Float> WithoutLapackData<F> for ViewRepr<&'a mut F::Lapack> {
    type D = ViewRepr<&'a mut F>;
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    #[cfg(feature = "ndarray-linalg")]
    use ndarray_linalg::eig::*;

    use super::{WithLapack, WithoutLapack};

    #[test]
    fn memory_check() {
        let a: Array2<f32> = Array2::zeros((20, 20));
        let a: Array2<f32> = a.with_lapack();

        assert_eq!(a.shape(), &[20, 20]);

        let b: Array2<f32> = a.clone().without_lapack();

        assert_eq!(b, a);
    }

    #[cfg(feature = "ndarray-linalg")]
    #[test]
    fn lapack_exists() {
        let a: Array2<f32> = Array2::zeros((4, 4));
        let a: Array2<f32> = a.with_lapack();

        let (_a, _b) = a.eig().unwrap();
    }
}
