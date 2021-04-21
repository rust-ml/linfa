use linfa::Float;
use ndarray::{ArrayView1, ArrayView2};

pub mod linear;

pub trait NearestNeighbour<'a, F: Float> {
    fn add_point(&mut self, point: ArrayView1<'a, F>);

    fn add_batch(&mut self, batch: &'a ArrayView2<'a, F>) {
        for row in batch.genrows() {
            self.add_point(row);
        }
    }

    fn reset(&mut self);

    fn num_points(&self) -> usize;

    // Returns nearest in order. Might want wrap in result or return iterator
    fn k_nearest(&self, point: ArrayView1<'a, F>, k: usize) -> Vec<ArrayView1<'a, F>>;

    fn within(&self, point: ArrayView1<'a, F>, range: F) -> Vec<ArrayView1<'a, F>>;
}
