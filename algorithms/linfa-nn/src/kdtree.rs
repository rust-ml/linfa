use std::marker::PhantomData;

use linfa::Float;
use ndarray::{aview1, Array2};
use ndarray_stats::DeviationExt;

use crate::{NearestNeighbour, NearestNeighbourBuilder, Point};

fn dist_fn<F: Float>(pt1: &[F], pt2: &[F]) -> F {
    aview1(pt1).sq_l2_dist(&aview1(pt2)).unwrap()
}

pub struct Kdtree<'a, F: Float>(kdtree::KdTree<F, Point<'a, F>, &'a [F]>);

impl<'a, F: Float> NearestNeighbour<'a, F> for Kdtree<'a, F> {
    fn from_batch(batch: &'a Array2<F>) -> Self {
        let mut tree = kdtree::KdTree::with_capacity(batch.ncols(), batch.nrows());
        for point in batch.genrows() {
            tree.add(point.to_slice().expect("views should be contiguous"), point)
                .unwrap();
        }
        Self(tree)
    }

    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Vec<Point<F>> {
        self.0
            .iter_nearest(
                point.to_slice().expect("views should ve contiguous"),
                &dist_fn,
            )
            .unwrap()
            .take(k)
            .map(|(_, pt)| pt.reborrow())
            .collect()
    }

    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Vec<Point<F>> {
        self.0
            .within(
                point.to_slice().expect("views should ve contiguous"),
                range,
                &dist_fn,
            )
            .unwrap()
            .into_iter()
            .map(|(_, pt)| pt.reborrow())
            .collect()
    }
}

pub struct KdtreeBuilder<F: Float>(PhantomData<F>);

impl<F: Float> NearestNeighbourBuilder<F> for KdtreeBuilder<F> {
    fn from_batch<'a>(&self, batch: &'a Array2<F>) -> Box<dyn 'a + NearestNeighbour<'a, F>> {
        Box::new(Kdtree::from_batch(batch))
    }
}
