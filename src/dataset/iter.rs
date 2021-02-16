use std::marker::PhantomData;
use super::{Float, Label, Records, AsTargets, DatasetView};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

pub struct Iter<'a, R: Records, T: AsTargets> {
    records: &'a R,
    targets: &'a T,
    idx: usize,
}

impl<'a, R: Records, T: AsTargets> Iter<'a, R, T> {
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

pub struct ChunksIter<'a, 'b: 'a, F: Float, E> {
    records: ArrayView2<'a, F>,
    targets: ArrayView2<'a, E>,
    size: usize,
    axis: Axis,
    idx: usize,
    phantom: PhantomData<&'b DatasetView<'b, F, E>>
}

impl<'a, 'b: 'a, F: Float, E> ChunksIter<'a, 'b, F, E> {
    pub fn new(records: ArrayView2<'a, F>, targets: ArrayView2<'a, E>, size: usize, axis: Axis) -> ChunksIter<'a, 'b, F, E> {
        ChunksIter {
            records,
            targets,
            size,
            axis,
            idx: 0,
            phantom: PhantomData
        }
    }
}

impl<'a, 'b: 'a, F: Float, E> Iterator for ChunksIter<'a, 'b, F, E> {
    type Item = DatasetView<'a, F, E>;

    fn next(&mut self) -> Option<DatasetView<'a, F, E>> {
        self.idx += 1;
        if self.idx == self.records.len_of(self.axis) / self.size {
            return None;
        }

        let (mut records, mut targets) = (self.records.reborrow(), self.targets.reborrow());
        records.slice_axis_inplace(self.axis, (self.idx*self.size..(self.idx+1)*self.size).into());
        targets.slice_axis_inplace(self.axis, (self.idx*self.size..(self.idx+1)*self.size).into());

        Some(DatasetView::from((records, targets)))
    }
}
