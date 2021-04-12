//! Implement Platt calibration with Newton method
//!

use crate::dataset::{AsTargets, DatasetBase, Pr, Records};
use crate::traits::{Fit, Predict, PredictRef};
use crate::Float;

use ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix1, Ix2};

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

impl<F: Float, O> Platt<F, O> {
    pub fn params() -> PlattParams<F> {
        PlattParams {
            maxiter: 100,
            minstep: F::from(1e-10).unwrap(),
            eps: F::from(1e-12).unwrap(),
        }
    }
}

impl<F: Float> PlattParams<F> {
    pub fn calibrate<'a, O, D, D2>(
        &self,
        obj: O,
        ds: DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D2, Ix1>>,
    ) -> Platt<F, O>
    where
        D: Data<Elem = F>,
        D2: Data<Elem = bool>,
        O: PredictRef<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>,
    {
        let predicted = obj.predict(&ds);

        let (a, b) = platt_newton_method(predicted.view(), ds.targets().view(), self);

        Platt { a, b, obj }
    }
}

impl<F: Float, D, O> PredictRef<ArrayBase<D, Ix2>, Array1<Pr>> for Platt<F, O>
where
    D: Data<Elem = F>,
    O: PredictRef<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>,
{
    fn predict_ref(&self, data: &ArrayBase<D, Ix2>) -> Array1<Pr> {
        self.obj
            .predict(data)
            .iter()
            .map(|x| {
                let f_apb = self.a * *x + self.b;
                let f_apb = f_apb.to_f32().unwrap();

                // avoid numerical problems for large f_apb
                if f_apb >= 0.0 {
                    Pr((-f_apb).exp() / (1.0 + (-f_apb).exp()))
                } else {
                    Pr(1.0 / (1.0 + f_apb.exp()))
                }
            })
            .collect()
    }
}

fn platt_newton_method<'a, F: Float>(
    reg_values: ArrayView1<'a, F>,
    labels: ArrayView1<'a, bool>,
    params: &PlattParams<F>,
) -> (F, F) {
    let (num_pos, num_neg) = labels.iter().fold((0, 0), |mut val, x| {
        match x {
            true => val.0 += 1,
            false => val.1 += 1,
        }

        val
    });
    let (num_pos, num_neg) = (num_pos as f32, num_neg as f32);

    let (hi_target, lo_target) = ((num_pos + 1.0) / (num_pos + 2.0), 1.0 / (num_neg + 2.0));

    let t = labels
        .iter()
        .map(|x| if *x { hi_target } else { lo_target })
        .map(|x| F::from(x).unwrap())
        .collect::<Vec<_>>();

    let reg_values = reg_values
        .into_iter()
        .map(|x| F::from(*x).unwrap())
        .collect::<Vec<_>>();

    let mut a = F::zero();
    let mut b = F::from((num_neg + 1.0) / (num_pos + 1.0)).unwrap().ln();
    let mut fval = F::zero();

    for (v, t) in reg_values.iter().zip(t.iter()) {
        let f_apb = *v * a + b;
        if f_apb >= F::zero() {
            fval += *t * f_apb + (F::one() + (-f_apb).exp()).ln();
        } else {
            fval += (*t - F::one()) * f_apb + (F::one() + f_apb.exp()).ln();
        }
    }

    for _ in 0..params.maxiter {
        let (mut h11, mut h22) = (params.eps, params.eps);
        let (mut h21, mut g1, mut g2) = (F::zero(), F::zero(), F::zero());

        for (v, t) in reg_values.iter().zip(t.iter()) {
            let f_apb = *v * a + b;

            let (p, q) = if f_apb >= F::zero() {
                (
                    (-f_apb).exp() / (F::one() + (-f_apb).exp()),
                    F::one() / (F::one() + (-f_apb).exp()),
                )
            } else {
                (
                    F::one() / (F::one() + f_apb.exp()),
                    f_apb.exp() / (F::one() + f_apb.exp()),
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

        dbg!(&g1, &g2);
        if g1.abs() < F::from(1e-5).unwrap() && g2.abs() < F::from(1e-5).unwrap() {
            break;
        }

        let det = h11 * h22 - h21 * h21;
        let d_a = -(h22 * g1 - h21 * g2) / det;
        let d_b = -(-h21 * g1 + h11 * g2) / det;
        let gd = g1 * d_a + g2 * d_b;

        //dbg!(&det, &d_a, &d_b, &gd);

        let mut stepsize = F::one();
        while stepsize >= params.minstep {
            let new_a = a + stepsize * d_a;
            let new_b = b + stepsize * d_b;
            let mut newf = F::zero();

            for (v, t) in reg_values.iter().zip(t.iter()) {
                let f_apb = *v * new_a + new_b;

                if f_apb >= F::zero() {
                    newf += *t * f_apb + (F::one() + (-f_apb).exp()).ln();
                } else {
                    newf += (*t - F::one()) * f_apb + (F::one() + f_apb.exp()).ln();
                }
            }

            if newf < fval + F::from(1e-4).unwrap() * stepsize * gd {
                a = new_a;
                b = new_b;
                fval = newf;
                break;
            } else {
                stepsize /= F::one() + F::one();
            }
        }

        if stepsize < params.minstep {
            //panic!("Line search failed!");
            break;
        }
    }

    (a, b)
}

#[cfg(test)]
mod tests {
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    use super::{platt_newton_method, PlattParams};
    use crate::Float;
    use ndarray::Array1;

    fn generate_dummy_values<F: Float, R: Rng>(
        a: F,
        b: F,
        n: usize,
        rng: &mut R,
    ) -> (Array1<F>, Array1<bool>) {
        // generate probability values, omit p = 0.0 to avoid infinity in reverse function
        let prob_values = Array1::linspace(F::one() / F::from(n).unwrap(), F::one(), n - 1);

        // generate regression values with inverse function
        let reg_values = prob_values
            .iter()
            .map(|x| (F::one() - *x) / *x)
            .map(|x| (x - b) / a)
            .collect();

        // roll decision according to probability
        let decisions = prob_values
            .iter()
            .map(|x| rng.gen_bool(x.to_f64().unwrap()))
            .collect();

        (reg_values, decisions)
    }

    #[test]
    fn newton_solver() {
        let mut rng = SmallRng::seed_from_u64(42);

        let testcases = &[
            (100_f32, 0.),
            /*(100., 0.),
            (10., 0.5),
            (100., 0.)*/
        ];

        let params = PlattParams {
            maxiter: 100,
            minstep: 1e-10,
            eps: 1e-12,
        };

        for (a, b) in testcases {
            let (reg_vals, dec_vals) = generate_dummy_values(*a, *b, 3000, &mut rng);
            let (a_est, b_est) = platt_newton_method(reg_vals.view(), dec_vals.view(), &params);
            dbg!(&a, &a_est, &b, &b_est);
        }
    }
}
