use std::{cmp::Reverse, collections::BinaryHeap, marker::PhantomData};

use linfa::Float;
use ndarray::{Array2, ArrayView2};
use noisy_float::NoisyFloat;

use crate::{
    distance::Distance, heap_elem::MinHeapElem, BuildError, NearestNeighbour,
    NearestNeighbourIndex, NnError, Point,
};

/// Spatial indexing structure created by [`LinearSearch`](struct.LinearSearch.html)
#[derive(Debug)]
pub struct LinearSearchIndex<'a, F: Float, D: Distance<F>>(ArrayView2<'a, F>, D);

impl<'a, F: Float, D: Distance<F>> LinearSearchIndex<'a, F, D> {
    /// Creates a new `LinearSearchIndex`
    pub fn new(batch: &'a Array2<F>, dist_fn: D) -> Result<Self, BuildError> {
        if batch.ncols() == 0 {
            Err(BuildError::ZeroDimension)
        } else {
            Ok(Self(batch.view(), dist_fn))
        }
    }
}

impl<'a, F: Float, D: Distance<F>> NearestNeighbourIndex<F> for LinearSearchIndex<'a, F, D> {
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Result<Vec<Point<F>>, NnError> {
        if self.0.ncols() != point.len() {
            Err(NnError::WrongDimension)
        } else {
            let mut heap = BinaryHeap::with_capacity(self.0.nrows());
            for pt in self.0.genrows() {
                let dist = self.1.rdistance(point.clone(), pt.clone());
                heap.push(MinHeapElem {
                    elem: pt.clone(),
                    dist: Reverse(NoisyFloat::new(dist)),
                });
            }

            Ok((0..k.min(heap.len()))
                .map(|_| heap.pop().unwrap().elem)
                .collect())
        }
    }

    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Result<Vec<Point<F>>, NnError> {
        if self.0.ncols() != point.len() {
            Err(NnError::WrongDimension)
        } else {
            let range = self.1.dist_to_rdist(range);
            Ok(self
                .0
                .genrows()
                .into_iter()
                .filter(|pt| self.1.rdistance(point.clone(), pt.clone()) < range)
                .collect())
        }
    }
}

/// Implementation of linear search, which is the simplest nearest neighbour algorithm. All queries
/// are implemented by scanning through every point, so all of them are `O(N)`.
#[derive(Default, Clone, Debug)]
pub struct LinearSearch<F: Float>(PhantomData<F>);

impl<F: Float> LinearSearch<F> {
    /// Creates an instance of `LinearSearch`
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: Float, D: 'static + Distance<F>> NearestNeighbour<F, D> for LinearSearch<F> {
    fn from_batch_with_leaf_size<'a>(
        &self,
        batch: &'a Array2<F>,
        leaf_size: usize,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbourIndex<F>>, BuildError> {
        if leaf_size == 0 {
            return Err(BuildError::EmptyLeaf);
        }
        LinearSearchIndex::new(batch, dist_fn)
            .map(|v| Box::new(v) as Box<dyn NearestNeighbourIndex<F>>)
    }
}
