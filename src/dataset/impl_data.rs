use ndarray::{ArrayBase, Axis, Dimension};

use super::{Data, Float};

impl<F: Float, S: ndarray::Data<Elem = F>, I: Dimension> Data for ArrayBase<S, I> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.len_of(Axis(0))
    }
}
