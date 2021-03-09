pub mod error;
mod linear_scaler;
mod norm_scaler;

pub use linear_scaler::{FittedLinearScaler, LinearScaler, ScalingMethod};
pub use norm_scaler::NormScaler;

pub trait Float: linfa::Float + ndarray_linalg::Lapack {}

impl Float for f32 {}
impl Float for f64 {}
