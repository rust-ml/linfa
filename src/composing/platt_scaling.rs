//! Implement Platt calibration with Newton method
//!

use crate::Float;
use crate::dataset::{AsTargets, DatasetBase, Pr, Records};
use crate::traits::Fit;

use ndarray::{ArrayBase, Ix2, Data, ArrayView1};

pub struct Platt<F, O> {
    a: F,
    b: F,
    obj: O,
}

pub struct PlattParams<F> {
    maxiter: usize,
    minstep: F,
    eps: F,
}

impl<'a, F: Float, O> Platt<F, O>
{
    pub fn params() -> PlattParams<F> {
        PlattParams {
            maxiter: 100,
            minstep: F::from(1e-10).unwrap(),
            eps: F::from(1e-12).unwrap(),
        }
    }
}

impl<F: Float, D, T> Fit<'_, ArrayBase<D, Ix2>, T> for PlattParams<F>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = bool>,
{
    type Object = Platt<F, O>;

    fn fit(&self, ds: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Self::Object {
        panic!("");
    }
}

fn platt_newton_method<'a, F: Float, O>(
    prob_values: ArrayView1<'a, Pr>,
    labels: ArrayView1<'a, bool>,
    num: (usize, usize),
    params: PlattParams<F>,
) -> Platt<F, O> {
    let (num_pos, num_neg) = labels.iter().fold((0, 0), |mut val, x| {
        match x {
            true => val.0 += 1,
            false => val.1 += 1
        }

        val
    });
    let (num_pos, num_neg) = (num_pos as f32, num_neg as f32);

    let (hi_target, lo_target) = (
        (num_pos + 1.0) / (num_pos + 2.0),
        1.0 / (num_neg + 2.0)
    );

    let t = labels.iter().map(|x| if *x { hi_target } else { lo_target })
        .map(|x| F::from(x).unwrap())
        .collect::<Vec<_>>();

    let prob_values = prob_values.into_iter().map(|x| F::from(x.0).unwrap())
        .collect::<Vec<_>>();

    let mut a = F::zero();
    let mut b = F::from((num_neg+1.0) / (num_pos + 1.0)).unwrap().ln();
    let mut fval = F::zero();

    for (v, t) in prob_values.iter().zip(t.iter()) {
        let f_apb = *v * a + b;
        if f_apb >= F::zero() {
            fval += *t * f_apb + (F::one() + -f_apb.exp()).ln();
        } else {
            fval += (*t - F::one()) * f_apb + (F::one() + f_apb.exp()).ln();
        }
    }

    for _ in 0..params.maxiter {
        let (mut h11, mut h22) = (params.eps, params.eps);
        let (mut h21, mut g1, mut g2) = (F::zero(), F::zero(), F::zero());

        for (v, t) in prob_values.iter().zip(t.iter()) {
            let f_apb = *v * a + b;

            let (p, q) = if f_apb >= F::zero() {
                (
                    (-f_apb).exp() / (F::one() + (-f_apb).exp()),
                    F::one() / (F::one() + (-f_apb).exp())
                )
            } else {
                (
                    F::one() / (F::one() + f_apb.exp()),
                    f_apb.exp() / (F::one() + f_apb.exp())
                )
            };

            let d2 = p * q;
            h11 += *v * *v * d2;
            h22 += d2;
            h21 += *v * d2;
            
            let d1 = *t - p;
            g1 += *v * d1;
            g2 += d1;
        }

        if g1.abs() < F::from(1e-5).unwrap() && g2.abs() < F::from(1e-5).unwrap() {
            break;
        }

        let det = h11 * h22 - h21 * h21;
        let d_a = (h22 * g1 - h21 * g2) / det;
        let d_b = -(-h21 * g1 + h11 * g2) / det;
        let gd = g1 * d_a + g2 * d_b;

        let mut stepsize = F::one();
        while stepsize >= params.minstep {
            let new_a = a + stepsize * d_a;
            let new_b = b + stepsize * d_b;
            let mut newf = F::zero();

            for (v, t) in prob_values.iter().zip(t.iter()) {
                let f_apb = *v * new_a + new_b;

                if f_apb >= F::zero() {
                    newf += *t * f_apb + (F::one() + (-f_apb).exp()).ln();
                } else {
                    newf += (*t - F::one()) * f_apb + (F::one() + (f_apb).exp()).ln();
                }
            }

            if newf < fval + F::from(1e-4).unwrap() * stepsize * gd {
                a = new_a;
                b = new_b;
                fval = newf;
            } else {
                stepsize /= F::one() + F::one();
            }
        }

        if stepsize < params.minstep {
            panic!("Line search failed!");
        }
    }

    Platt { 
        a, 
        b,
        obj: params.obh,
    }
}

