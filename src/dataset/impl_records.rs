use ndarray::{ArrayBase, Axis, Dimension, Data};

use super::{Records, Float, Dataset};

impl<F: Float, S: Data<Elem = F>, I: Dimension> Records for ArrayBase<S, I> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.len_of(Axis(0))
    }
}

impl<F: Float, S: Data<Elem = F>, I: Dimension> Records for &ArrayBase<S, I> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.len_of(Axis(0))
    }
}
impl<F: Float, D: Records<Elem = F>, T> Records for Dataset<D, T> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.records.observations()
    }
}

impl<F: Float, D: Records<Elem = F>, T> Records for &Dataset<D, T> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.records.observations()
    }
}
