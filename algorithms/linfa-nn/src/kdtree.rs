use std::marker::PhantomData;

use linfa::Float;
use ndarray::{aview1, Array2};

use crate::{
    distance::{CommonDistance, Distance},
    BuildError, NearestNeighbour, NearestNeighbourBuilder, NnError, Point,
};

pub struct KdTree<'a, F: Float, D: Distance<F> = CommonDistance<F>>(
    kdtree::KdTree<F, Point<'a, F>, &'a [F]>,
    D,
);

impl<'a, F: Float, D: Distance<F>> KdTree<'a, F, D> {
    pub fn from_batch(batch: &'a Array2<F>, dist_fn: D) -> Result<Self, BuildError> {
        if batch.ncols() == 0 {
            Err(BuildError::ZeroDimension)
        } else {
            let mut tree =
                kdtree::KdTree::with_capacity(batch.ncols().max(1), batch.nrows().max(1));
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

impl<'a, F: Float, D: Distance<F>> NearestNeighbour<F> for KdTree<'a, F, D> {
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Result<Vec<Point<F>>, NnError> {
        Ok(self
            .0
            .iter_nearest(
                point.to_slice().expect("views should be contiguous"),
                &|a, b| self.1.distance(aview1(a), aview1(b)),
            )?
            .take(k)
            .map(|(_, pt)| pt.reborrow())
            .collect())
    }

    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Result<Vec<Point<F>>, NnError> {
        Ok(self
            .0
            .within(
                point.to_slice().expect("views should be contiguous"),
                range,
                &|a, b| self.1.distance(aview1(a), aview1(b)),
            )?
            .into_iter()
            .map(|(_, pt)| pt.reborrow())
            .collect())
    }
}

#[derive(Default)]
pub struct KdTreeBuilder<F: Float>(PhantomData<F>);

impl<F: Float, D: 'static + Distance<F>> NearestNeighbourBuilder<F, D> for KdTreeBuilder<F> {
    fn from_batch<'a>(
        &self,
        batch: &'a Array2<F>,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbour<F>>, BuildError> {
        KdTree::from_batch(batch, dist_fn).map(|v| Box::new(v) as Box<dyn NearestNeighbour<F>>)
    }
}
