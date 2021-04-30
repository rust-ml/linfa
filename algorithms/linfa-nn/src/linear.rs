use std::{cmp::Reverse, collections::BinaryHeap, marker::PhantomData};

use linfa::Float;
use ndarray::{Array2, ArrayView2};
use noisy_float::NoisyFloat;

use crate::{
    distance::{CommonDistance, Distance},
    heap_elem::HeapElem,
    NearestNeighbour, NearestNeighbourBuilder, Point,
};

pub struct LinearSearch<'a, F: Float, D: Distance<F> = CommonDistance<F>>(ArrayView2<'a, F>, D);

type HeapPoint<'a, F> = HeapElem<F, Point<'a, F>>;

impl<'a, F: Float, D: Distance<F>> LinearSearch<'a, F, D> {
    fn from_batch(batch: &'a Array2<F>, dist_fn: D) -> Self {
        Self(batch.view(), dist_fn)
    }
}

impl<'a, F: Float, D: Distance<F>> NearestNeighbour<F> for LinearSearch<'a, F, D> {
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Vec<Point<F>> {
        let mut heap = BinaryHeap::with_capacity(self.0.nrows());
        for pt in self.0.genrows() {
            let dist = self.1.distance(point.clone(), pt.clone());
            heap.push(HeapPoint {
                elem: pt.clone(),
                dist: Reverse(NoisyFloat::new(dist)),
            });
        }
        (0..k.min(heap.len()))
            .map(|_| heap.pop().unwrap().elem)
            .collect()
    }

    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Vec<Point<F>> {
        self.0
            .genrows()
            .into_iter()
            .filter(|pt| self.1.distance(point.clone(), pt.clone()) < range)
            .collect()
    }
}

#[derive(Default)]
pub struct LinearSearchBuilder<F: Float>(PhantomData<F>);

impl<F: Float, D: 'static + Distance<F>> NearestNeighbourBuilder<F, D> for LinearSearchBuilder<F> {
    fn from_batch<'a>(
        &self,
        batch: &'a Array2<F>,
        dist_fn: D,
    ) -> Box<dyn 'a + NearestNeighbour<F>> {
        Box::new(LinearSearch::from_batch(batch, dist_fn))
    }
}
