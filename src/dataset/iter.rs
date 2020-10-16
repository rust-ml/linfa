use super::{Float, Label, Records, Targets};
use ndarray::{s, Array2, ArrayView1};

pub struct Iter<'a, R: Records, T: Targets> {
    records: &'a R,
    targets: &'a T,
    idx: usize,
}

impl<'a, R: Records, T: Targets> Iter<'a, R, T> {
    pub fn new(records: &'a R, targets: &'a T) -> Iter<'a, R, T> {
        Iter {
            records,
            targets,
            idx: 0,
        }
    }
}

impl<'a, F: Float, L: Label> Iterator for Iter<'a, Array2<F>, Vec<L>> {
    type Item = (ArrayView1<'a, F>, &'a L);

    fn next(&mut self) -> Option<Self::Item> {
        if self.records.observations() >= self.idx {
            return None;
        }

        Some((
            self.records.slice(s![self.idx, ..]),
            &self.targets[self.idx],
        ))
    }
}
