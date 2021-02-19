pub mod error;
mod float;
pub mod glm;
pub mod ols;

pub use error::Result;
pub use glm::TweedieRegressor;
pub use ols::LinearRegression;
