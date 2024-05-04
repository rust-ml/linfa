mod adaboost;
mod random_forest;
// mod random_forest_regressor;
mod gradient_boost;

pub use adaboost::*;
pub use random_forest::*;
// pub use random_forest_regressor::*;
pub use gradient_boost::*;

pub use linfa::error::Result;
