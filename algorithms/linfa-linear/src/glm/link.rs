//! Link functions used by GLM

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::float::Float;

#[derive(Copy, Clone, Serialize, Deserialize)]
/// Link functions used by GLM
pub enum Link {
    /// The identity link function `g(x)=x`
    Identity,
    /// The log link function `g(x)=log(x)`
    Log,
    /// The logit link function `g(x)=logit(x)`
    Logit,
}

impl Link {
    /// Compute the link function `g(ypred)`
    ///
    /// The link function links the mean `ypred=E[y]` to the so called
    /// linear predictor, `g(ypred)=linear predictor`
    pub fn link<A: Float>(&self, ypred: &Array1<A>) -> Array1<A> {
        match self {
            Self::Identity => IdentityLink::link(ypred),
            Self::Log => LogLink::link(ypred),
            Self::Logit => LogitLink::link(ypred),
        }
    }

    /// Computes the derivative of the link `g'(ypred)`
    pub fn link_derivative<A: Float>(&self, ypred: &Array1<A>) -> Array1<A> {
        match self {
            Self::Identity => IdentityLink::link_derivative(ypred),
            Self::Log => LogLink::link_derivative(ypred),
            Self::Logit => LogitLink::link_derivative(ypred),
        }
    }

    /// Computes the inverse link function `h(linear predictor)`
    ///
    /// Gives the inverse relationship between the linear predictor and the mean
    /// `ypred=E[y]`, i.e. `h(linear predictor)=ypred`
    pub fn inverse<A: Float>(&self, lin_pred: &Array1<A>) -> Array1<A> {
        match self {
            Self::Identity => IdentityLink::inverse(lin_pred),
            Self::Log => LogLink::inverse(lin_pred),
            Self::Logit => LogitLink::inverse(lin_pred),
        }
    }

    /// Computes the derivative of the inverse link function `h'(linear predictor)`
    pub fn inverse_derviative<A: Float>(&self, lin_pred: &Array1<A>) -> Array1<A> {
        match self {
            Self::Identity => IdentityLink::inverse_derivative(lin_pred),
            Self::Log => LogLink::inverse_derivative(lin_pred),
            Self::Logit => LogitLink::inverse_derivative(lin_pred),
        }
    }
}

trait LinkFn<A> {
    fn link(ypred: &Array1<A>) -> Array1<A>;
    fn link_derivative(ypred: &Array1<A>) -> Array1<A>;
    fn inverse(lin_pred: &Array1<A>) -> Array1<A>;
    fn inverse_derivative(lin_pred: &Array1<A>) -> Array1<A>;
}

struct IdentityLink;

impl<A: Float> LinkFn<A> for IdentityLink {
    fn link(ypred: &Array1<A>) -> Array1<A> {
        ypred.clone()
    }

    fn link_derivative(ypred: &Array1<A>) -> Array1<A> {
        Array1::ones(ypred.shape()[0])
    }

    fn inverse(lin_pred: &Array1<A>) -> Array1<A> {
        lin_pred.clone()
    }

    fn inverse_derivative(lin_pred: &Array1<A>) -> Array1<A> {
        Array1::ones(lin_pred.shape()[0])
    }
}

struct LogLink;

impl<A: Float> LinkFn<A> for LogLink {
    fn link(ypred: &Array1<A>) -> Array1<A> {
        ypred.mapv(|x| num_traits::Float::ln(x))
    }

    fn link_derivative(ypred: &Array1<A>) -> Array1<A> {
        // 1 / ypred
        ypred.mapv(|x| {
            let lower_bound = A::from(1e-7).unwrap();
            if x < lower_bound {
                return lower_bound.recip();
            }
            x.recip()
        })
    }

    fn inverse(lin_pred: &Array1<A>) -> Array1<A> {
        lin_pred.mapv(|x| num_traits::Float::exp(x))
    }

    fn inverse_derivative(lin_pred: &Array1<A>) -> Array1<A> {
        lin_pred.mapv(|x| num_traits::Float::exp(x))
    }
}

struct LogitLink;

impl<A: Float> LinkFn<A> for LogitLink {
    fn link(ypred: &Array1<A>) -> Array1<A> {
        // logit(ypred)
        ypred.mapv(|x| num_traits::Float::ln(x / (A::from(1.).unwrap() - x)))
    }

    fn link_derivative(ypred: &Array1<A>) -> Array1<A> {
        // 1 / (ypred * (1-ypred)
        ypred.mapv(|x| A::from(1.).unwrap() / (x * (A::from(1.).unwrap() - x)))
    }

    fn inverse(lin_pred: &Array1<A>) -> Array1<A> {
        // expit(lin_pred)
        lin_pred.mapv(|x| {
            A::from(1.).unwrap() / (A::from(1.).unwrap() + num_traits::Float::exp(x.neg()))
        })
    }

    fn inverse_derivative(lin_pred: &Array1<A>) -> Array1<A> {
        // expit(lin_pred) * (1 - expit(lin_pred))
        let expit = lin_pred.mapv(|x| {
            A::from(1.).unwrap() / (A::from(1.).unwrap() + num_traits::Float::exp(x.neg()))
        });
        let one_minus_expit = expit.mapv(|x| A::from(1.).unwrap() - x);
        expit * one_minus_expit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    macro_rules! test_links {
        ($($func:ident: {input: $input:expr, expected: $expected:expr, link: $link:expr}),*) => {
            $(
                #[test]
                fn $func() {
                    for (expected, input) in $expected.iter().zip($input.iter()) {
                        let output = $link(input);
                        assert_abs_diff_eq!(output, expected, epsilon = 1e-6);
                    }
                }
            )*
        };
    }

    test_links! [
        test_identity_link: {
            input: vec![array![1., 1., 1., 1.], array![1.348, 2.879, 4.545, 3.232]],
            expected: vec![array![1., 1., 1., 1.], array![1.348, 2.879, 4.545, 3.232]],
            link: IdentityLink::link
        },
        test_identity_link_derivative: {
            input: vec![array![1., 1., 1., 1.], array![1.348, 2.879, 4.545, 3.232]],
            expected: vec![array![1., 1., 1., 1.], array![1., 1., 1., 1.]],
            link: IdentityLink::link_derivative
        },
        test_identity_inverse: {
            input: vec![array![1., 1., 1., 1.], array![1.348, 2.879, 4.545, 3.232]],
            expected: vec![array![1., 1., 1., 1.], array![1.348, 2.879, 4.545, 3.232]],
            link: IdentityLink::inverse
        },
        test_identity_inverse_derivative: {
            input: vec![array![1., 1., 1., 1.], array![1.348, 2.879, 4.545, 3.232]],
            expected: vec![array![1., 1., 1., 1.], array![1., 1., 1., 1.]],
            link: IdentityLink::inverse_derivative
        }
    ];

    test_links! [
        test_log_link: {
            input: vec![
                array![1.382, 1.329, 1.32, 1.322],
                array![4.56432e+01, 4.30000e+01, 2.00000e-07, 3.42000e-01],
            ],
            expected: vec![
                array![0.32353173, 0.28442678, 0.27763174, 0.27914574],
                array![3.82085464, 3.76120012, -15.42494847, -1.07294454],
            ],
            link: LogLink::link
        },
        test_log_link_derivative: {
            input: vec![
                array![1.382, 1.329, 1.32, 1.322],
                array![4.56432e+01, 4.30000e+01, 2.00000e-07, 3.42000e-01],
            ],
            expected:vec![
                array![0.723589, 0.75244545, 0.75757576, 0.75642965],
                array![
                    2.19090686e-02,
                    2.32558140e-02,
                    5.00000000e+06,
                    2.92397661e+00
                ],
            ],
            link: LogLink::link_derivative
        },
        test_log_inverse: {
            input: vec![
                array![1.382f32, 1.329f32, 1.32f32, 1.322f32],
                array![4.56432e+01, 4.30000e+01, 2.00000e-07, 3.42000e-01],
            ],
            expected: vec![
                array![3.98285939, 3.77726423, 3.74342138, 3.75091571],
                array![6.646452e+19, 4.72783947e+18, 1.00000020e+00, 1.40776030e+00],
            ],
            link: LogLink::inverse
        },
        test_log_inverse_derivative: {
            input: vec![
                array![1.382f32, 1.329f32, 1.32f32, 1.322f32],
                array![4.56432e+01, 4.30000e+01, 2.00000e-07, 3.42000e-01],
            ],
            expected: vec![
                array![3.98285939, 3.77726423, 3.74342138, 3.75091571],
                array![6.646452e+19, 4.72783947e+18, 1.00000020e+00, 1.40776030e+00],
            ],
            link: LogLink::inverse_derivative
        }
    ];

    test_links! [
        test_logit_link: {
            input: vec![
                array![0.934, 0.323, 0.989, 0.412], array![0.044, 0.023, 0.999, 0.124]
            ],
            expected: vec![
                array![2.6498217, -0.74001895, 4.49879906, -0.3557036 ],
                array![-3.07856828, -3.74899244,  6.90675478, -1.95508453],
            ],
            link: LogitLink::link
        },
        test_logit_link_derivative: {
            input: vec![array![0.934, 0.323, 0.989, 0.412], array![0.044, 0.023, 0.999, 0.124]],
            expected: vec![
                array![16.22217896, 4.57308011, 91.92021325, 4.12786474],
                array![23.77329783, 44.50180232, 1001.001001, 9.20606864],
            ],
            link: LogitLink::link_derivative
        },
        test_logit_inverse: {
            input: vec![array![0.934, 0.323, 0.989, 0.412], array![0.044, 0.023, 0.999, 0.124]],
            expected: vec![
                array![0.71788609, 0.5800552, 0.72889036, 0.60156734],
                array![0.51099823, 0.50574975, 0.73086192, 0.53096034],
            ],
            link: LogitLink::inverse
        },
        test_logit_inverse_derivative: {
            input: vec![array![0.934, 0.323, 0.989, 0.412], array![0.044, 0.023, 0.999, 0.124]],
            expected: vec![
                array![0.20252565, 0.24359116, 0.1976092, 0.23968407],
                array![0.24987904, 0.24996694, 0.19670277, 0.24904146],
            ],
            link: LogitLink::inverse_derivative
        }
    ];
}
