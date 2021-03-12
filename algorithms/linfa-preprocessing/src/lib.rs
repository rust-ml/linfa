mod count_vectorizer;
pub mod error;
mod linear_scaler;
mod norm_scaler;

pub use count_vectorizer::{CountVectorizer, FittedCountVectorizer};
pub use linear_scaler::{FittedLinearScaler, LinearScaler, ScalingMethod};
pub use norm_scaler::NormScaler;

use approx;

pub trait Float: linfa::Float + ndarray_linalg::Lapack + approx::AbsDiffEq {}

impl Float for f32 {}
impl Float for f64 {}
