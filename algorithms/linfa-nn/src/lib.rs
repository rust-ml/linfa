use linfa::Float;
use ndarray::{Array2, ArrayView1};

pub mod kdtree;
pub mod linear;

pub(crate) type Point<'a, F> = ArrayView1<'a, F>;

pub trait NearestNeighbour<'a, F: Float> {
    fn from_batch(batch: &'a Array2<F>) -> Self
    where
        Self: Sized;

    // Returns nearest in order. Might want wrap in result or return iterator
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Vec<Point<F>>;

    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Vec<Point<F>>;
}

pub trait NearestNeighbourBuilder<F: Float> {
    fn from_batch<'a>(&self, batch: &'a Array2<F>) -> Box<dyn 'a + NearestNeighbour<'a, F>>;
}

#[cfg(test)]
mod test {
    use ndarray::arr2;

    use super::*;

    fn nn_test(builder: Box<dyn NearestNeighbourBuilder<f64>>) {
        let points = arr2(&[[0.0, 2.0], [10.0, 4.0], [4.0, 5.0]]);
        {
            let nn = builder.from_batch(&points);

            let p = points.row(0);
            let out = nn.k_nearest(p, 5);
            drop(out);
        }
    }
}
