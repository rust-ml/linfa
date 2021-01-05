use super::{DatasetBase, Float, Records, Targets};
use ndarray::{ArrayBase, Axis, Data, Dimension};

/// Implement records for NdArrays
impl<F: Float, S: Data<Elem = F>, I: Dimension> Records for ArrayBase<S, I> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.len_of(Axis(0))
    }
}

/// Implement records for a DatasetBase
impl<F: Float, D: Records<Elem = F>, T: Targets> Records for DatasetBase<D, T> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.records.observations()
    }
}

/// Implement records for an empty dataset
impl Records for () {
    type Elem = ();

    fn observations(&self) -> usize {
        0
    }
}

/// Implement records for references
impl<R: Records> Records for &R {
    type Elem = R::Elem;

    fn observations(&self) -> usize {
        (*self).observations()
    }
}
