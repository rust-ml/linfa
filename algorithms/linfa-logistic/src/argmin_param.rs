//! This module defines newtypes for ndarray's Array.
//!
//! This is necessary to be able to abstract over floats (f32 and f64) so that
//! the logistic regression code can be abstract in the float type it works
//! with.
//!
//! Unfortunately, this requires that we re-implement some traits from Argmin.

use crate::float::Float;
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminL1Norm, ArgminL2Norm, ArgminMinMax, ArgminMul, ArgminSignum,
    ArgminSub, ArgminZeroLike,
};
use ndarray::{Array, ArrayBase, Data, Dimension, Ix1, Ix2, Zip};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

pub fn elem_dot<F: linfa::Float, A1: Data<Elem = F>, A2: Data<Elem = F>, D: Dimension>(
    a: &ArrayBase<A1, D>,
    b: &ArrayBase<A2, D>,
) -> F {
    Zip::from(a)
        .and(b)
        .fold(F::zero(), |acc, &a, &b| acc + a * b)
}

#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct ArgminParam<F, D: Dimension>(pub Array<F, D>);

impl<F, D: Dimension> ArgminParam<F, D> {
    #[inline]
    pub fn as_array(&self) -> &Array<F, D> {
        &self.0
    }
}

impl<F: Float, D: Dimension> ArgminSub<F, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn sub(&self, other: &F) -> ArgminParam<F, D> {
        ArgminParam(&self.0 - *other)
    }
}

impl<F: Float, D: Dimension> ArgminSub<ArgminParam<F, D>, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn sub(&self, other: &ArgminParam<F, D>) -> ArgminParam<F, D> {
        ArgminParam(&self.0 - &other.0)
    }
}

impl<F: Float, D: Dimension> ArgminAdd<F, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn add(&self, other: &F) -> ArgminParam<F, D> {
        ArgminParam(&self.0 + *other)
    }
}

impl<F: Float, D: Dimension> ArgminAdd<ArgminParam<F, D>, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn add(&self, other: &ArgminParam<F, D>) -> ArgminParam<F, D> {
        ArgminParam(&self.0 + &other.0)
    }
}

impl<F: Float, D: Dimension> ArgminDot<ArgminParam<F, D>, F> for ArgminParam<F, D> {
    fn dot(&self, other: &ArgminParam<F, D>) -> F {
        elem_dot(&self.0, &other.0)
    }
}

impl<F: Float, D: Dimension> ArgminL1Norm<F> for ArgminParam<F, D> {
    fn l1_norm(&self) -> F {
        num_traits::Float::sqrt(elem_dot(&self.0, &self.0))
    }
}

impl<F: Float, D: Dimension> ArgminL2Norm<F> for ArgminParam<F, D> {
    fn l2_norm(&self) -> F {
        num_traits::Float::sqrt(elem_dot(&self.0, &self.0))
    }
}

impl<F: Float, D: Dimension> ArgminMul<F, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn mul(&self, other: &F) -> ArgminParam<F, D> {
        ArgminParam(&self.0 * *other)
    }
}

impl<F: Float, D: Dimension> ArgminMul<ArgminParam<F, D>, ArgminParam<F, D>> for ArgminParam<F, D> {
    fn mul(&self, other: &ArgminParam<F, D>) -> ArgminParam<F, D> {
        ArgminParam(&self.0 * &other.0)
    }
}

impl<F: Float, D: Dimension> ArgminSignum for ArgminParam<F, D> {
    fn signum(self) -> ArgminParam<F, D> {
        self
    }
}

impl<F: Float, D: Dimension> ArgminZeroLike for ArgminParam<F, D> {
    fn zero_like(&self) -> ArgminParam<F, D> {
        let dims = self.as_array().raw_dim();
        ArgminParam(Array::zeros(dims))
    }
}

impl<F: Float> ArgminMinMax for ArgminParam<F, Ix1> {
    fn min(x: &Self, y: &Self) -> ArgminParam<F, Ix1> {
        let x_array = x.as_array();
        let y_array = y.as_array();

        assert_eq!(x_array.shape(), y_array.shape());
        ArgminParam(
            x_array
                .iter()
                .zip(y_array)
                .map(|(&a, &b)| if a < b { a } else { b })
                .collect(),
        )
    }

    fn max(x: &Self, y: &Self) -> ArgminParam<F, Ix1> {
        let x_array = x.as_array();
        let y_array = y.as_array();

        assert_eq!(x_array.shape(), y_array.shape());
        ArgminParam(
            x_array
                .iter()
                .zip(y_array)
                .map(|(&a, &b)| if a > b { a } else { b })
                .collect(),
        )
    }
}

impl<F: Float> ArgminMinMax for ArgminParam<F, Ix2> {
    fn min(x: &Self, y: &Self) -> ArgminParam<F, Ix2> {
        let x_array = x.as_array();
        let y_array = y.as_array();

        assert_eq!(x_array.shape(), y_array.shape());
        let m = x_array.shape()[0];
        let n = x_array.shape()[1];
        let mut out = x_array.clone();
        for i in 0..m {
            for j in 0..n {
                let a = x_array[(i, j)];
                let b = y_array[(i, j)];
                out[(i, j)] = if a < b { a } else { b };
            }
        }
        ArgminParam(out)
    }

    fn max(x: &Self, y: &Self) -> ArgminParam<F, Ix2> {
        let x_array = x.as_array();
        let y_array = y.as_array();

        assert_eq!(x_array.shape(), y_array.shape());
        let m = x_array.shape()[0];
        let n = x_array.shape()[1];
        let mut out = x_array.clone();
        for i in 0..m {
            for j in 0..n {
                let a = x_array[(i, j)];
                let b = y_array[(i, j)];
                out[(i, j)] = if a > b { a } else { b };
            }
        }
        ArgminParam(out)
    }
}
