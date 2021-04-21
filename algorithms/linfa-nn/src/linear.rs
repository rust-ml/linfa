use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    marker::PhantomData,
};

use linfa::Float;
use ndarray::{Array2, ArrayView2};
use ndarray_stats::DeviationExt;
use ordered_float::NotNan;

use crate::{NearestNeighbour, NearestNeighbourBuilder, Point};

fn dist_fn<F: Float>(pt1: &Point<F>, pt2: &Point<F>) -> F {
    pt1.sq_l2_dist(&pt2).unwrap()
}

pub struct LinearSearch<'a, F: Float>(ArrayView2<'a, F>);

struct HeapElem<'a, F: Float> {
    dist: Reverse<NotNan<F>>,
    point: Point<'a, F>,
}

impl<'a, F: Float> PartialEq for HeapElem<'a, F> {
    fn eq(&self, other: &Self) -> bool {
        self.dist.eq(&other.dist)
    }
}
impl<'a, F: Float> Eq for HeapElem<'a, F> {}

impl<'a, F: Float> PartialOrd for HeapElem<'a, F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl<'a, F: Float> Ord for HeapElem<'a, F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}

impl<'a, F: Float> NearestNeighbour<'a, F> for LinearSearch<'a, F> {
    fn from_batch(batch: &'a Array2<F>) -> Self {
        Self(batch.view())
    }

    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Vec<Point<F>> {
        let mut heap = BinaryHeap::with_capacity(self.0.nrows());
        for pt in self.0.genrows() {
            let dist = dist_fn(&point, &pt);
            heap.push(HeapElem {
                point: pt.clone(),
                dist: Reverse(NotNan::new(dist).expect("distance should not be NaN")),
            });
        }
        (0..k).map(|_| heap.pop().unwrap().point).collect()
    }

    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Vec<Point<F>> {
        self.0
            .genrows()
            .into_iter()
            .filter(|pt| dist_fn(&point, &pt) < range)
            .collect()
    }
}

pub struct LinearSearchBuilder<F: Float>(PhantomData<F>);

impl<F: Float> NearestNeighbourBuilder<F> for LinearSearchBuilder<F> {
    fn from_batch<'a>(&self, batch: &'a Array2<F>) -> Box<dyn 'a + NearestNeighbour<'a, F>> {
        Box::new(LinearSearch::from_batch(batch))
    }
}
