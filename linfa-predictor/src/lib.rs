use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

/// Trait every predictor should implement
pub trait Predictor {
    /// predict class for each sample
    fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<u64>;
    /// predict probability of each possible class for each sample
    fn predict_proba(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Vec<Array1<f64>>;
}
