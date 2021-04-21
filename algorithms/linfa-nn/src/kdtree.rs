use linfa::Float;
use ndarray::aview1;
use ndarray_stats::DeviationExt;

use crate::{NearestNeighbour, Point};

fn dist_fn<F: Float>(pt1: &[F], pt2: &[F]) -> F {
    aview1(pt1).sq_l2_dist(&aview1(pt2)).unwrap()
}

pub struct Kdtree<'a, F: Float>(kdtree::KdTree<F, Point<'a, F>, &'a [F]>);

impl<'a, F: Float> NearestNeighbour<'a, F> for Kdtree<'a, F> {
    fn add_point(&mut self, point: Point<'a, F>) {
        self.0
            .add(point.to_slice().expect("views should be contiguous"), point)
            .unwrap();
    }

    fn num_points(&self) -> usize {
        self.0.size()
    }

    fn k_nearest(&self, point: Point<'a, F>, k: usize) -> Vec<Point<'a, F>> {
        self.0
            .iter_nearest(
                point.to_slice().expect("views should ve contiguous"),
                &dist_fn,
            )
            .unwrap()
            .take(k)
            .map(|(_, pt)| pt)
            .cloned()
            .collect()
    }

    fn within_range(&self, point: Point<'a, F>, range: F) -> Vec<Point<'a, F>> {
        self.0
            .within(
                point.to_slice().expect("views should ve contiguous"),
                range,
                &dist_fn,
            )
            .unwrap()
            .into_iter()
            .map(|(_, pt)| pt)
            .cloned()
            .collect()
    }
}
