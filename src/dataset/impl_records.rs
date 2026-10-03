use super::{DatasetBase, Records};
use ndarray::{ArrayBase, Axis, Data, Dimension};
use sprs::{CsMatBase, SpIndex};
use std::ops::Deref;

/// Implement records for NdArrays
impl<F, S: Data<Elem = F>, I: Dimension> Records for ArrayBase<S, I> {
    type Elem = F;

    fn nsamples(&self) -> usize {
        self.len_of(Axis(0))
    }

    fn nfeatures(&self) -> usize {
        self.len_of(Axis(1))
    }
}

/// Implement records for sparse (CSR/CSC) matrices, both owned and views
impl<N, I, Iptr, IpS, IS, DS> Records for CsMatBase<N, I, IpS, IS, DS, Iptr>
where
    I: SpIndex,
    Iptr: SpIndex,
    IpS: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [N]>,
{
    type Elem = N;

    fn nsamples(&self) -> usize {
        self.rows()
    }

    fn nfeatures(&self) -> usize {
        self.cols()
    }
}

/// Implement records for a DatasetBase
impl<F, D: Records<Elem = F>, T> Records for DatasetBase<D, T> {
    type Elem = F;

    fn nsamples(&self) -> usize {
        self.records.nsamples()
    }

    fn nfeatures(&self) -> usize {
        self.records.nfeatures()
    }
}

/// Implement records for an empty dataset
impl Records for () {
    type Elem = ();

    fn nsamples(&self) -> usize {
        0
    }

    fn nfeatures(&self) -> usize {
        0
    }
}

/// Implement records for references
impl<R: Records> Records for &R {
    type Elem = R::Elem;

    fn nsamples(&self) -> usize {
        (*self).nsamples()
    }

    fn nfeatures(&self) -> usize {
        (*self).nfeatures()
    }
}
