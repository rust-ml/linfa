use super::{Float, Label, Dataset, Data, Targets};
use ndarray::{Array2, ArrayView1, s};

pub struct Iter<'a, D: Data, T: Targets> {
    data: &'a D,
    targets: &'a T,
    idx: usize
}

impl<'a, D: Data, T: Targets> Iter<'a, D, T> {
    pub fn new(data: &'a D, targets: &'a T) -> Iter<'a, D, T> {
        Iter {
            data,
            targets,
            idx: 0
        }
    }
}

impl<'a, F: Float, L: Label> Iterator for Iter<'a, Array2<F>, Vec<L>> {
    type Item = (ArrayView1<'a, F>, &'a L);

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.observations() >= self.idx {
            return None;
        }

        Some((self.data.slice(s![self.idx, ..]), &self.targets[self.idx]))
    }
}
