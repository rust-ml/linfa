use ndarray::Array2;
use super::{Float, Label, Dataset, iter::Iter};

impl<F: Float, L: Label> Dataset<Array2<F>, Vec<L>> {
    pub fn iter<'a>(&'a self) -> Iter<'a, Array2<F>, Vec<L>> {
        Iter::new(&self.data, &self.targets)
    }
}
