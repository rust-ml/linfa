use std::marker::PhantomData;

use linfa::Float;
use ndarray::{aview1, Array2};

use crate::{
    distance::{CommonDistance, Distance},
    BuildError, NearestNeighbour, NearestNeighbourIndex, NnError, Point,
};

pub struct KdTreeIndex<'a, F: Float, D: Distance<F> = CommonDistance<F>>(
    kdtree::KdTree<F, Point<'a, F>, &'a [F]>,
    D,
);

impl<'a, F: Float, D: Distance<F>> KdTreeIndex<'a, F, D> {
    pub fn new(batch: &'a Array2<F>, leaf_size: usize, dist_fn: D) -> Result<Self, BuildError> {
        if batch.ncols() == 0 {
            Err(BuildError::ZeroDimension)
        } else {
            let mut tree = kdtree::KdTree::with_capacity(batch.ncols().max(1), leaf_size);
            for point in batch.genrows() {
                tree.add(point.to_slice().expect("views should be contiguous"), point)
                    .unwrap();
            }
            Ok(Self(tree, dist_fn))
        }
    }
}

impl From<kdtree::ErrorKind> for NnError {
    fn from(err: kdtree::ErrorKind) -> Self {
        match err {
            kdtree::ErrorKind::WrongDimension => NnError::WrongDimension,
            kdtree::ErrorKind::NonFiniteCoordinate => panic!("infinite value found"),
            _ => unreachable!(),
        }
    }
}

impl<'a, F: Float, D: Distance<F>> NearestNeighbourIndex<F> for KdTreeIndex<'a, F, D> {
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Result<Vec<Point<F>>, NnError> {
        Ok(self
            .0
            .nearest(
                point.to_slice().expect("views should be contiguous"),
                k,
                &|a, b| self.1.rdistance(aview1(a), aview1(b)),
            )?
            .into_iter()
            .map(|(_, pt)| pt.reborrow())
            .collect())
    }

    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Result<Vec<Point<F>>, NnError> {
        let range = self.1.dist_to_rdist(range);
        Ok(self
            .0
            .within(
                point.to_slice().expect("views should be contiguous"),
                range,
                &|a, b| self.1.rdistance(aview1(a), aview1(b)),
            )?
            .into_iter()
            .map(|(_, pt)| pt.reborrow())
            .collect())
    }
}

#[derive(Default)]
pub struct KdTree<F: Float>(PhantomData<F>);

impl<F: Float> KdTree<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: Float, D: 'static + Distance<F>> NearestNeighbour<F, D> for KdTree<F> {
    fn from_batch_with_leaf_size<'a>(
        &self,
        batch: &'a Array2<F>,
        leaf_size: usize,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbourIndex<F>>, BuildError> {
        KdTreeIndex::new(batch, leaf_size, dist_fn)
            .map(|v| Box::new(v) as Box<dyn NearestNeighbourIndex<F>>)
    }
}
