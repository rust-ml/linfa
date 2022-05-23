use super::{AsTargets, DatasetBase, DatasetView, FromTargetArray, Records, TargetDim};
use ndarray::{ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, Data, Ix2};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Iter<'a, 'b: 'a, F, L, I> {
    records: ArrayView2<'b, F>,
    targets: ArrayView<'b, L, I>,
    idx: usize,
    phantom: PhantomData<&'a ArrayView2<'b, F>>,
}

impl<'a, 'b: 'a, F, L, I> Iter<'a, 'b, F, L, I> {
    pub fn new(records: ArrayView2<'b, F>, targets: ArrayView<'b, L, I>) -> Iter<'a, 'b, F, L, I> {
        Iter {
            records,
            targets,
            idx: 0,
            phantom: PhantomData,
        }
    }
}

impl<'a, 'b: 'a, F, L, I: TargetDim> Iterator for Iter<'a, 'b, F, L, I> {
    type Item = (ArrayView1<'a, F>, ArrayView<'a, L, I::Smaller>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.records.nsamples() <= self.idx {
            return None;
        }

        self.idx += 1;
        let records = self.records.reborrow();
        let targets = self.targets.clone().reborrow();

        Some((
            records.index_axis_move(Axis(0), self.idx - 1),
            targets.index_axis_move(Axis(0), self.idx - 1),
        ))
    }
}

#[derive(Clone)]
pub struct DatasetIter<'a, 'b, R: Records, T> {
    dataset: &'b DatasetBase<R, T>,
    idx: usize,
    target_or_feature: bool,
    phantom: PhantomData<&'a ArrayView2<'a, R::Elem>>,
}

impl<'a, 'b: 'a, R: Records, T> DatasetIter<'a, 'b, R, T> {
    pub fn new(
        dataset: &'b DatasetBase<R, T>,
        target_or_feature: bool,
    ) -> DatasetIter<'a, 'b, R, T> {
        DatasetIter {
            dataset,
            idx: 0,
            target_or_feature,
            phantom: PhantomData,
        }
    }
}

impl<'a, 'b: 'a, F: 'a, L: 'a, D, T> Iterator for DatasetIter<'a, 'b, ArrayBase<D, Ix2>, T>
where
    D: Data<Elem = F>,
    T: FromTargetArray<'a, Elem = L>,
{
    type Item = DatasetView<'a, F, L, T::Ix>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.target_or_feature && self.dataset.ntargets() <= self.idx {
            return None;
        }

        if self.target_or_feature && self.dataset.nfeatures() <= self.idx {
            return None;
        }

        let mut records = self.dataset.records.view();
        let mut targets = self.dataset.targets.as_targets();
        let feature_names;
        let weights = self.dataset.weights.clone();

        if !self.target_or_feature {
            // This branch should only run for 2D targets
            targets.collapse_axis(Axis(1), self.idx);
            feature_names = self.dataset.feature_names.clone();
        } else {
            records.collapse_axis(Axis(1), self.idx);
            if self.dataset.feature_names.len() == records.len_of(Axis(1)) {
                feature_names = vec![self.dataset.feature_names[self.idx].clone()];
            } else {
                feature_names = Vec::new();
            }
        }

        self.idx += 1;

        let dataset_view = DatasetBase {
            records,
            targets,
            weights,
            feature_names,
        };

        Some(dataset_view)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChunksIter<'a, 'b: 'a, F, T> {
    records: ArrayView2<'a, F>,
    targets: &'a T,
    size: usize,
    axis: Axis,
    idx: usize,
    phantom: PhantomData<&'b ArrayView2<'a, F>>,
}

impl<'a, 'b: 'a, F, T> ChunksIter<'a, 'b, F, T> {
    pub fn new(
        records: ArrayView2<'a, F>,
        targets: &'a T,
        size: usize,
        axis: Axis,
    ) -> ChunksIter<'a, 'b, F, T> {
        ChunksIter {
            records,
            targets,
            size,
            axis,
            idx: 0,
            phantom: PhantomData,
        }
    }
}

impl<'a, 'b: 'a, F, E: 'b, T> Iterator for ChunksIter<'a, 'b, F, T>
where
    T: AsTargets<Elem = E> + FromTargetArray<'b>,
{
    type Item = DatasetBase<ArrayView2<'a, F>, T::View>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.records.len_of(self.axis) / self.size {
            return None;
        }

        let (mut records, mut targets) = (
            self.records.reborrow(),
            self.targets.as_targets().reborrow(),
        );
        records.slice_axis_inplace(
            self.axis,
            (self.idx * self.size..(self.idx + 1) * self.size).into(),
        );
        targets.slice_axis_inplace(
            self.axis,
            (self.idx * self.size..(self.idx + 1) * self.size).into(),
        );

        self.idx += 1;

        Some(DatasetBase::new(records, T::new_targets_view(targets)))
    }
}
