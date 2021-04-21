use linfa::Float;
use ndarray::{ArrayView1, ArrayView2};

pub mod kdtree;
pub mod linear;

pub(crate) type Point<'a, F> = ArrayView1<'a, F>;

pub trait NearestNeighbour<'a, F: Float> {
    fn add_point(&mut self, point: Point<'a, F>);

    fn add_batch(&mut self, batch: &'a ArrayView2<'a, F>) {
        for row in batch.genrows() {
            self.add_point(row);
        }
    }

    fn num_points(&self) -> usize;

    // Returns nearest in order. Might want wrap in result or return iterator
    fn k_nearest(&self, point: Point<'a, F>, k: usize) -> Vec<Point<'a, F>>;

    fn within_range(&self, point: Point<'a, F>, range: F) -> Vec<Point<'a, F>>;
}
