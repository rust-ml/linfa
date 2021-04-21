use linfa::Float;
use ndarray::{ArrayView1, ArrayView2};

pub mod kdtree;
pub mod linear;

pub(crate) type Point<'a, F> = ArrayView1<'a, F>;

pub trait NearestNeighbour<'a, F: Float> {
    fn from_batch(batch: &'a ArrayView2<'a, F>) -> Self
    where
        Self: Sized;

    // Returns nearest in order. Might want wrap in result or return iterator
    fn k_nearest(&'a self, point: Point<'a, F>, k: usize) -> Vec<Point<'a, F>>;

    fn within_range(&'a self, point: Point<'a, F>, range: F) -> Vec<Point<'a, F>>;
}

pub trait NearestNeighbourBuilder<F: Float> {
    fn from_batch<'a>(&self, batch: &'a ArrayView2<'a, F>)
        -> Box<dyn 'a + NearestNeighbour<'a, F>>;
}
