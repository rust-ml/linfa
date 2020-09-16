use ndarray::{Array1, ArrayBase, Data, Ix2};

/// Trait every predictor should implement
pub trait Predictor {
    fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<u64>;
}
