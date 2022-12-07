#![doc = include_str!("../README.md")]

pub mod dataset;
#[cfg(feature = "generate")]
pub mod generate;

pub use dataset::*;
