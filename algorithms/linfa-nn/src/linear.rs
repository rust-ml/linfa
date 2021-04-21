use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

use linfa::Float;
use ndarray_stats::DeviationExt;
use ordered_float::NotNan;

use crate::{NearestNeighbour, Point};

fn dist_fn<F: Float>(pt1: &Point<F>, pt2: &Point<F>) -> F {
    pt1.sq_l2_dist(&pt2).unwrap()
}

pub struct LinearSearch<'a, F: Float>(Vec<Point<'a, F>>);

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
    fn add_point(&mut self, point: Point<'a, F>) {
        self.0.push(point);
    }

    fn num_points(&self) -> usize {
        self.0.len()
    }

    fn k_nearest(&self, point: Point<'a, F>, k: usize) -> Vec<Point<'a, F>> {
        let mut heap = BinaryHeap::with_capacity(self.num_points());
        for pt in self.0.iter() {
            let dist = dist_fn(&point, &pt);
            heap.push(HeapElem {
                point: pt.clone(),
                dist: Reverse(NotNan::new(dist).expect("distance should not be NaN")),
            });
        }
        (0..k).map(|_| heap.pop().unwrap().point).collect()
    }

    fn within_range(&self, point: Point<'a, F>, range: F) -> Vec<Point<'a, F>> {
        self.0
            .iter()
            .filter(|pt| dist_fn(&point, &pt) < range)
            .cloned()
            .collect()
    }
}
