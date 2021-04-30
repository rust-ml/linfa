use std::{cmp::Reverse, collections::BinaryHeap, marker::PhantomData};

use linfa::Float;
use ndarray::{Array2, ArrayView2};
use ndarray_stats::DeviationExt;
use noisy_float::NoisyFloat;

use crate::{heap_elem::HeapElem, NearestNeighbour, NearestNeighbourBuilder, Point};

fn dist_fn<F: Float>(pt1: &Point<F>, pt2: &Point<F>) -> F {
    pt1.sq_l2_dist(&pt2).unwrap()
}

pub struct LinearSearch<'a, F: Float>(ArrayView2<'a, F>);

type HeapPoint<'a, F> = HeapElem<F, Point<'a, F>>;

impl<'a, F: Float> LinearSearch<'a, F> {
    fn from_batch(batch: &'a Array2<F>) -> Self {
        Self(batch.view())
    }
}

impl<'a, F: Float> NearestNeighbour<F> for LinearSearch<'a, F> {
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Vec<Point<F>> {
        let mut heap = BinaryHeap::with_capacity(self.0.nrows());
        for pt in self.0.genrows() {
            let dist = dist_fn(&point, &pt);
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
            .filter(|pt| dist_fn(&point, &pt) < range)
            .collect()
    }
}

#[derive(Default)]
pub struct LinearSearchBuilder<F: Float>(PhantomData<F>);

impl<F: Float> NearestNeighbourBuilder<F> for LinearSearchBuilder<F> {
    fn from_batch<'a>(&self, batch: &'a Array2<F>) -> Box<dyn 'a + NearestNeighbour<F>> {
        Box::new(LinearSearch::from_batch(batch))
    }
}
