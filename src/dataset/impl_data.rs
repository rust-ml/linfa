use ndarray::{ArrayBase, Axis, Dimension};

use super::{Data, Float, Dataset};

impl<F: Float, S: ndarray::Data<Elem = F>, I: Dimension> Data for ArrayBase<S, I> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.len_of(Axis(0))
    }
}

impl<F: Float, D: Data<Elem = F>, T> Data for Dataset<D, T> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.data.observations()
    }
}
