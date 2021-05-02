use std::marker::PhantomData;

use linfa::Float;
use ndarray::{aview1, Array2};

use crate::{
    distance::{CommonDistance, Distance},
    NearestNeighbour, NearestNeighbourBuilder, Point,
};

pub struct KdTree<'a, F: Float, D: Distance<F> = CommonDistance<F>>(
    kdtree::KdTree<F, Point<'a, F>, &'a [F]>,
    D,
);

impl<'a, F: Float, D: Distance<F>> KdTree<'a, F, D> {
    pub fn from_batch(batch: &'a Array2<F>, dist_fn: D) -> Self {
        let mut tree = kdtree::KdTree::with_capacity(batch.ncols().max(1), batch.nrows().max(1));
        for point in batch.genrows() {
            tree.add(point.to_slice().expect("views should be contiguous"), point)
                .unwrap();
        }
        Self(tree, dist_fn)
    }
}

impl<'a, F: Float, D: Distance<F>> NearestNeighbour<F> for KdTree<'a, F, D> {
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Vec<Point<F>> {
        self.0
            .iter_nearest(
                point.to_slice().expect("views should ve contiguous"),
                &|a, b| self.1.distance(aview1(a), aview1(b)),
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
                &|a, b| self.1.distance(aview1(a), aview1(b)),
            )
            .unwrap()
            .into_iter()
            .map(|(_, pt)| pt.reborrow())
            .collect()
    }
}

#[derive(Default)]
pub struct KdTreeBuilder<F: Float>(PhantomData<F>);

impl<F: Float, D: 'static + Distance<F>> NearestNeighbourBuilder<F, D> for KdTreeBuilder<F> {
    fn from_batch<'a>(
        &self,
        batch: &'a Array2<F>,
        dist_fn: D,
    ) -> Box<dyn 'a + NearestNeighbour<F>> {
        Box::new(KdTree::from_batch(batch, dist_fn))
    }
}
