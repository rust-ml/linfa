use crate::Float;
use ndarray::{Data, OwnedRepr, ViewRepr, ArrayBase, Dimension};

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

pub trait WithLapackData
where
    Self: Data
{
    type D: Data;

    /// Add trait bound `Lapack` and `Scalar` to NdArray's floating point
    ///
    /// This is safe, because only implemented for D == Self
    fn with_lapack<I>(x: ArrayBase<Self, I>) -> ArrayBase<Self::D, I>
    where
        I: Dimension,
    {
        unsafe {
            std::ptr::read(x.as_ptr() as *const ArrayBase<Self::D, I>)
        }
    }
}

impl<F: Float> WithLapackData for OwnedRepr<F> {
    type D = OwnedRepr<F::Lapack>;
}

impl<'a, F: Float> WithLapackData for ViewRepr<&'a F> {
    type D = ViewRepr<&'a F::Lapack>;
}

pub trait WithoutLapackData<F: Float>
where
    Self: Data
{
    type D: Data<Elem = F>;

    /// Add trait bound `Lapack` and `Scalar` to NdArray's floating point
    ///
    /// This is safe, because only implemented for D == Self
    fn without_lapack<I>(x: ArrayBase<Self, I>) -> ArrayBase<Self::D, I>
    where
        I: Dimension,
    {
        unsafe {
            std::ptr::read(x.as_ptr() as *const ArrayBase<Self::D, I>)
        }
    }
}

impl<F: Float> WithoutLapackData<F> for OwnedRepr<F::Lapack> {
    type D = OwnedRepr<F>;
}

impl<'a, F: Float> WithoutLapackData<F> for ViewRepr<&'a F::Lapack> {
    type D = ViewRepr<&'a F>;
}
