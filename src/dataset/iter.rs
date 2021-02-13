use super::{Float, Label, Records, ToTargets, DatasetView};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, iter::AxisChunksIter, Ix2, Axis};

pub struct Iter<'a, R: Records, T: ToTargets> {
    records: &'a R,
    targets: &'a T,
    idx: usize,
}

impl<'a, R: Records, T: ToTargets> Iter<'a, R, T> {
    pub fn new(records: &'a R, targets: &'a T) -> Iter<'a, R, T> {
        Iter {
            records,
            targets,
            idx: 0,
        }
    }
}

impl<'a, F: Float, L: Label> Iterator for Iter<'a, Array2<F>, Array1<L>> {
    type Item = (ArrayView1<'a, F>, &'a L);

    fn next(&mut self) -> Option<Self::Item> {
        if self.records.nsamples() >= self.idx {
            return None;
        }

        Some((
            self.records.slice(s![self.idx, ..]),
            self.targets.get(self.idx).unwrap(),
        ))
    }
}

pub struct ChunksIter<'a, F: Float, E> {
    records: ArrayView2<'a, F>,
    targets: ArrayView2<'a, E>,
    size: usize,
}

impl<'a, F: Float, E> ChunksIter<'a, F, E> {
    pub fn new(records: ArrayView2<'a, F>, targets: ArrayView2<'a, E>, size: usize) -> ChunksIter<'a, F, E> {
        ChunksIter {
            records,
            targets,
            size,
        }
    }
}

impl<'a, F: Float, E> Iterator for ChunksIter<'a, F, E> {
    type Item = DatasetView<'a, F, E>;

    fn next(&mut self) -> Option<Self::Item> {
        panic!("")
    }
}
