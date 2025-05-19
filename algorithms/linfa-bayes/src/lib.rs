#![doc = include_str!("../README.md")]

mod base_nb;
mod bernoulli_nb;
mod error;
mod gaussian_nb;
mod hyperparams;
mod multinomial_nb;

pub use base_nb::NaiveBayes;
pub use bernoulli_nb::BernoulliNb;
pub use error::{NaiveBayesError, Result};
pub use gaussian_nb::GaussianNb;
pub use hyperparams::{BernoulliNbParams, BernoulliNbValidParams};
pub use hyperparams::{GaussianNbParams, GaussianNbValidParams};
pub use hyperparams::{MultinomialNbParams, MultinomialNbValidParams};
pub use multinomial_nb::MultinomialNb;

use linfa::{Float, Label};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// Histogram of class occurrences for multinomial and binomial parameter estimation
#[derive(Debug, Default, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub(crate) struct ClassHistogram<F> {
    class_count: usize,
    prior: F,
    feature_count: Array1<F>,
    feature_log_prob: Array1<F>,
}

impl<F: Float> ClassHistogram<F> {
    // Update log probabilities of features given class
    fn update_with_smoothing(&mut self, x_new: ArrayView2<F>, alpha: F, total_count: bool) {
        // If incoming data is empty no updates required
        if x_new.nrows() == 0 {
            return;
        }

        // unpack old class information
        let ClassHistogram {
            class_count,
            feature_count,
            feature_log_prob,
            ..
        } = self;

        // count new feature occurrences
        let feature_count_new: Array1<F> = x_new.sum_axis(Axis(0));

        // if previous batch was empty, we send the new feature count calculated
        if *class_count > 0 {
            *feature_count = feature_count_new + feature_count.view();
        } else {
            *feature_count = feature_count_new;
        }

        // apply smoothing to feature counts
        let feature_count_smoothed = feature_count.mapv(|x| x + alpha);

        // compute total count (smoothed)
        let count = if total_count {
            F::cast(x_new.nrows()) + alpha * F::cast(2)
        } else {
            feature_count_smoothed.sum()
        };

        // compute log probabilities of each feature
        *feature_log_prob = feature_count_smoothed.mapv(|x| x.ln() - count.ln());
        // update class count
        *class_count += x_new.nrows();
    }
}

/// Returns a subset of x corresponding to the class specified by `ycondition`
pub(crate) fn filter<F: Float, L: Label + Ord>(
    x: ArrayView2<F>,
    y: ArrayView1<L>,
    ycondition: &L,
) -> Array2<F> {
    // We identify the row numbers corresponding to the class we are interested in
    let index = y
        .into_iter()
        .enumerate()
        .filter(|&(_, y)| (*ycondition == *y))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    // We subset x to only records corresponding to the class represented in `ycondition`
    let mut xsubset = Array2::zeros((index.len(), x.ncols()));
    index
        .into_iter()
        .enumerate()
        .for_each(|(i, r)| xsubset.row_mut(i).assign(&x.slice(s![r, ..])));

    xsubset
}
