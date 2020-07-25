use crate::float::Float;
use argmin::prelude::*;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Clone, Deserialize, Debug, Default)]
pub struct ArgminParam<F>(pub Array1<F>);

impl<F> ArgminParam<F> {
    #[inline]
    pub fn as_array(&self) -> &Array1<F> {
        &self.0
    }
}

impl<F: Float> ArgminSub<ArgminParam<F>, ArgminParam<F>> for ArgminParam<F> {
    fn sub(&self, other: &ArgminParam<F>) -> ArgminParam<F> {
        ArgminParam(&self.0 - &other.0)
    }
}

impl<F: Float> ArgminAdd<ArgminParam<F>, ArgminParam<F>> for ArgminParam<F> {
    fn add(&self, other: &ArgminParam<F>) -> ArgminParam<F> {
        ArgminParam(&self.0 + &other.0)
    }
}

impl<F: Float> ArgminDot<ArgminParam<F>, FloatWrapper<F>> for ArgminParam<F> {
    fn dot(&self, other: &ArgminParam<F>) -> FloatWrapper<F> {
        FloatWrapper(self.0.dot(&other.0))
    }
}

impl<F: Float> ArgminNorm<FloatWrapper<F>> for ArgminParam<F> {
    fn norm(&self) -> FloatWrapper<F> {
        FloatWrapper(self.0.dot(&self.0))
    }
}

impl<F: Float> ArgminMul<FloatWrapper<F>, ArgminParam<F>> for ArgminParam<F> {
    fn mul(&self, other: &FloatWrapper<F>) -> ArgminParam<F> {
        ArgminParam(&self.0 * other.0)
    }
}

impl<F: Float> ArgminMul<ArgminParam<F>, ArgminParam<F>> for FloatWrapper<F> {
    fn mul(&self, other: &ArgminParam<F>) -> ArgminParam<F> {
        ArgminParam(&other.0 * self.0)
    }
}

impl<F: Float> ArgminMul<ArgminParam<F>, ArgminParam<F>> for ArgminParam<F> {
    fn mul(&self, other: &ArgminParam<F>) -> ArgminParam<F> {
        ArgminParam(&self.0 * &other.0)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct FloatWrapper<F>(pub F);

impl<F: Float> std::fmt::Display for FloatWrapper<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        std::fmt::Display::fmt(&self.0, formatter)
    }
}

impl<F: Float> std::ops::Rem for FloatWrapper<F> {
    type Output = FloatWrapper<F>;
    fn rem(self, rhs: Self) -> Self::Output {
        FloatWrapper(self.0.rem(rhs.0))
    }
}
impl<F: Float> std::ops::Div for FloatWrapper<F> {
    type Output = FloatWrapper<F>;
    fn div(self, rhs: Self) -> Self::Output {
        FloatWrapper(self.0.div(rhs.0))
    }
}
impl<F: Float> std::ops::Add for FloatWrapper<F> {
    type Output = FloatWrapper<F>;
    fn add(self, rhs: Self) -> Self::Output {
        FloatWrapper(self.0.add(rhs.0))
    }
}
impl<F: Float> std::ops::Mul for FloatWrapper<F> {
    type Output = FloatWrapper<F>;
    fn mul(self, rhs: Self) -> Self::Output {
        FloatWrapper(self.0.mul(rhs.0))
    }
}
impl<F: Float> std::ops::Sub for FloatWrapper<F> {
    type Output = FloatWrapper<F>;
    fn sub(self, rhs: Self) -> Self::Output {
        FloatWrapper(self.0.sub(rhs.0))
    }
}
impl<F: Float> std::ops::Neg for FloatWrapper<F> {
    type Output = FloatWrapper<F>;
    fn neg(self) -> Self::Output {
        FloatWrapper(self.0.neg())
    }
}

impl<F: Float> num_traits::identities::One for FloatWrapper<F> {
    fn one() -> Self {
        FloatWrapper(F::one())
    }
}
impl<F: Float> num_traits::identities::Zero for FloatWrapper<F> {
    fn zero() -> Self {
        FloatWrapper(F::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl<F: Float> num_traits::cast::ToPrimitive for FloatWrapper<F> {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
}
impl<F: Float> num_traits::cast::NumCast for FloatWrapper<F> {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(|x| FloatWrapper(x))
    }
}
impl<F: Float> num_traits::cast::FromPrimitive for FloatWrapper<F> {
    fn from_i64(n: i64) -> Option<Self> {
        F::from_i64(n).map(|x| FloatWrapper(x))
    }
    fn from_u64(n: u64) -> Option<Self> {
        F::from_u64(n).map(|x| FloatWrapper(x))
    }
}

impl<F: Float> num_traits::Num for FloatWrapper<F> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(str, radix).map(|x| FloatWrapper(x))
    }
}

/// Generate a method implementation that forwards the call to the wrapped
/// float variable and potentially wrap the return value in a FloatWrapper
/// Multiple methods can be chained, separated by `;`
macro_rules! forward {
    ($( $method:ident ( self $( , $arg:ident : $ty:tt )* ) -> $ret:tt ; )*)
        => {$(
            #[inline]
            fn $method(self $( , $arg : $ty )* ) -> $ret {
                wrap_return!($ret, <F as num_traits::Float>::$method(self.0 $( , unwrap_arg!($arg: $ty) )* ))
            }
        )*};
}

/// Conditionally wrap the return type in a FloatWrapper if it is of type
/// `Self`. This is a helper macro for `forward`.
macro_rules! wrap_return {
    (Self, $expr:expr) => {
        FloatWrapper($expr)
    };
    ($ty:ty, $expr:expr) => {
        $expr
    };
}

/// Conditionally unwrap an argument by accessing `.0` if it is of type
/// `Self`. This is a helper macro for `forward`.
macro_rules! unwrap_arg {
    ($arg:ident: Self) => {
        $arg.0
    };
    ($arg:ident: $ty:tt) => {
        $arg
    };
}

/// Given a list of method names separated by `,`, generate static method
/// implementations that forward the calls to the wrapped float type and
/// wrap the return values in a FloatWrapper.
macro_rules! forward_static {
    ($($method:ident,)* )
        => {$(
            #[inline]
            fn $method() -> Self {
                FloatWrapper(F::$method())
            }
        )*};
}

/// Given a list of method names separated by `,`, generate query method
/// implementations that forward the calls to the wrapped float value.
macro_rules! forward_query {
    ($($method:ident,)* )
        => {$(
            #[inline]
            fn $method(self) -> bool {
                F::$method(self.0)
            }
        )*};
}

/// Given a list of method names separated by `,`, generate method
/// implementations that forward the calls to the wrapped float type and
/// wrap the return values in a FloatWrapper.
macro_rules! forward_wrapped {
    ($($method:ident,)*)
        => {$(
            #[inline]
            fn $method(self) -> Self {
                FloatWrapper(<F as num_traits::Float>::$method(self.0))
            }
        )*};
}

impl<F: Float> num_traits::Float for FloatWrapper<F> {
    forward_static!(
        nan,
        infinity,
        neg_infinity,
        neg_zero,
        min_value,
        min_positive_value,
        max_value,
    );
    forward_query!(
        is_nan,
        is_infinite,
        is_finite,
        is_normal,
        is_sign_positive,
        is_sign_negative,
    );
    forward_wrapped!(
        floor, ceil, round, trunc, fract, abs, signum, recip, sqrt, exp, exp2, ln, log2, log10,
        cbrt, sin, cos, tan, asin, acos, atan, exp_m1, ln_1p, sinh, cosh, tanh, asinh, acosh,
        atanh,
    );
    forward!(
        mul_add(self, a: Self, b: Self) -> Self;
        powi(self, n: i32) -> Self;
        powf(self, n: Self) -> Self;
        log(self, base: Self) -> Self;
        max(self, other: Self) -> Self;
        min(self, other: Self) -> Self;
        abs_sub(self, other: Self) -> Self;
        hypot(self, other: Self) -> Self;
        atan2(self, other: Self) -> Self;
    );
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.0.sin_cos();
        (FloatWrapper(sin), FloatWrapper(cos))
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.integer_decode()
    }
    fn classify(self) -> std::num::FpCategory {
        self.0.classify()
    }
}

impl<F: Float> FloatConst for FloatWrapper<F> {
    forward_static!(
        E,
        FRAC_1_PI,
        FRAC_1_SQRT_2,
        FRAC_2_PI,
        FRAC_2_SQRT_PI,
        FRAC_PI_2,
        FRAC_PI_3,
        FRAC_PI_4,
        FRAC_PI_6,
        FRAC_PI_8,
        LN_10,
        LN_2,
        LOG10_E,
        LOG2_E,
        PI,
        SQRT_2,
    );
}
