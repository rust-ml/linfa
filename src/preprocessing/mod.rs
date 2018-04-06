use ndarray::prelude::*;
use ndarray::Array2;

use std::ops::{Add, Mul, Div};
use libnum::{Num, FromPrimitive, pow};

fn increamental_mean_and_var<T>(x: &Array2<T>, last_mean: T, last_variance: T, last_sample_count: usize)
    where T: Num + Clone + FromPrimitive
{
    let last_sum = last_mean * FromPrimitive::from_usize(last_sample_count).unwrap();
    let new_sum = x.sum_axis(Axis(0));

    let new_sample_count = x.len_of(Axis(0));
    let updated_sample_count = last_sample_count + new_sample_count;

    let updated_mean = (last_sum + new_sum) / (FromPrimitive::from_usize(updated_sample_count).unwrap());

    // We need a function to compute the variance!
    let new_unnormalized_variance = x.var_axis(Axis(0)) * FromPrimitive::from_usize(new_sample_count);
    if last_sample_count == 0 {
        let updated_unnormalized_variance = new_unnormalized_variance;
    } else {
        let last_over_new_count = (last_sample_count as f64) / (new_sample_count as f64);
        let last_unnormalized_variance = last_variance * FromPrimitive::from_usize(last_sample_count).unwrap();
        let updated_unnormalized_variance = (
            last_unnormalized_variance +
            new_unnormalized_variance +
            FromPrimitive::from_f64(last_over_new_count).unwrap() /
			FromPrimitive::from_usize(updated_sample_count).unwrap() *
            pow(last_sum / FromPrimitive::from_f64(last_over_new_count).unwrap() - new_sum, 2)
        );
        let updated_variance = updated_unnormalized_variance / FromPrimitive::from_usize(updated_sample_count).unwrap();
    }
}
