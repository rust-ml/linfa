//! # Independent Component Analysis (ICA)
//!
//! `linfa-ica` aims to provide pure Rust implementations of ICA algorithms.
//!
//! ICA separates mutivariate signals into their additive, independent subcomponents.
//! ICA is primarily used for separating superimposed signals and not for dimensionality
//! reduction.
//!
//! Input data is whitened (remove underlying correlation) before modeling.
//!
//! ## The Big Picture
//!
//! `linfa-ica` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem,
//! an effort to create a toolkit for classical Machine Learning implemented in pure Rust,
//! akin to Python's `scikit-learn`.
//!
//! ## Current state
//!
//! `linfa-ica` currently provides an implementation of the following methods:
//!
//! - Fast Independent Component Analysis (Fast ICA)
//!
//! ## Example
//!
//! Here's an example of ICA unmixing the mixture of two signals
//!
//! ```
//! use linfa::{
//!     dataset::DatasetBase,
//!     traits::{Fit, Predict},
//! };
//! use linfa_ica::fast_ica::{FastIca, GFunc};
//! use ndarray::{array, concatenate};
//! use ndarray::{Array, Array2, Axis};
//! use ndarray_npy::write_npy;
//! use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
//! use rand_isaac::Isaac64Rng;
//!
//! let nsamples = 2000;
//! // Creating a sine wave signal
//! let source1 = Array::linspace(0., 8., nsamples).mapv(|x| (2f64 * x).sin());
//! // Creating a sawtooth signal
//! let source2 = Array::linspace(0., 8., nsamples).mapv(|x| {
//!     let tmp = (4f64 * x).sin();
//!     if tmp > 0. {
//!         return 1.;
//!     }
//!     -1.
//! });
//!
//! // Column concatenating both the signals
//! let mut sources_original = concatenate![
//!     Axis(1),
//!     source1.insert_axis(Axis(1)),
//!     source2.insert_axis(Axis(1))
//! ];
//!
//! // Adding random noise to the signals
//! let mut rng = Isaac64Rng::seed_from_u64(42);
//! sources_original +=
//!     &Array::random_using((2000, 2), Uniform::new(0.0, 1.0), &mut rng).mapv(|x| x * 0.2);
//!
//! // Mixing the two signals
//! let mixing = array![[1., 1.], [0.5, 2.]];
//! // Shape of the data is (2000 x 2)
//! // This data will be unmixed by ICA to recover back the original signals
//! let sources_mixed = sources_original.dot(&mixing.t());
//!
//! // Fitting the model
//! // We set the G function used in the approximation of neg-entropy as logcosh
//! // with its alpha value as 1
//! // `ncomponents` is not set, it will be automatically be assigned 2 from
//! // the input
//! let ica = FastIca::new().gfunc(GFunc::Logcosh(1.0));
//! let ica = ica.fit(&DatasetBase::from(sources_mixed.view())).unwrap();
//!
//! // Here we unmix the data to recover back the original signals
//! let sources_ica = ica.predict(&sources_mixed);
//! ```

#[macro_use]
extern crate ndarray;

pub mod error;
pub mod fast_ica;
