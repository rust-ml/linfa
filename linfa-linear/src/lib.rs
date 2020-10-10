pub mod elastic;
pub mod error;
mod float;
pub mod glm;
pub mod ols;

pub use glm::TweedieRegressor;
pub use ols::LinearRegression;
pub use elastic::ElasticNet;
