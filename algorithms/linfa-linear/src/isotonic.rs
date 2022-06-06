//! Isotonic
#![allow(non_snake_case)]
use crate::error::{LinearError, Result};
use ndarray::{s, stack, Array1, ArrayBase, Axis, Data, Ix1, Ix2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use linfa::dataset::{AsSingleTargets, DatasetBase};
use linfa::traits::{Fit, PredictInplace};

pub trait Float: linfa::Float {}
impl Float for f32 {}
impl Float for f64 {}

// Isotonic regression (IR) algorithms
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PVA;
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct AlgorithmA;

// IR trait
pub trait IR {
    fn evaluate<F, D>(
        &self,
        y: &ArrayBase<D, Ix1>,
        weights: Option<&[f32]>,
        index: &Vec<usize>,
    ) -> (Vec<F>, Vec<usize>)
    where
        F: Float,
        D: Data<Elem = F>;
}

/// An implementation of algorithm A from Best (1990)
/// for solving IRC problem
impl IR for AlgorithmA {
    fn evaluate<F, D>(
        &self,
        ys: &ArrayBase<D, Ix1>,
        weights: Option<&[f32]>,
        index: &Vec<usize>,
    ) -> (Vec<F>, Vec<usize>)
    where
        F: Float,
        D: Data<Elem = F>,
    {
        let n = ys.len();

        // Precompute partial averages
        let mut AvU = vec![F::zero(); n + 1];
        let mut wsum = F::zero();
        for i in 1..=n {
            let ii = n - i;
            let ui = index[ii];
            let w = if weights.is_none() {
                F::one()
            } else {
                F::cast(weights.unwrap()[ui])
            };
            AvU[ii] = (ys[ui] * w + AvU[ii + 1] * wsum) / (wsum + w);
            wsum += w;
        }
        //println!("y:{:?}\nAvU:{:?}", y, AvU);

        let mut V = Vec::<F>::new();
        let mut W = Vec::<F>::new();
        let mut J_index = vec![0_usize; n];
        J_index[0] = n - 1;
        let mut i = 0; // B0
        let mut j = 1; // Uj
        while i + j < n {
            let AvB0 = AvU[i];
            //println!("AvB0: {AvB0}, i:{i}, j:{j}, AvUj:{}", AvU[i + j]);
            //println!("i:{i} j:{j}, V:{:?} W:{:?}, J:{:?}", V, W, J_index);
            if AvU[i + j] <= AvB0 {
                // Step 1
                j += 1;
            } else {
                // Step 2
                let B_minus_index = i + j - 1;
                J_index[i] = B_minus_index; // B- = L_j
                J_index[B_minus_index] = i;
                J_index[B_minus_index + 1] = n - 1; // B0 = U_j
                J_index[n - 1] = B_minus_index + 1;

                // Step 2.1
                let (mut AvB_minus, mut B_minus_w) = waverage(&ys, weights, i, J_index[i], &index);

                let mut P_B_minus_index = i;
                let AvP_B_minus = *V.last().unwrap_or(&F::neg_infinity());
                while V.len() > 0 && AvB_minus < AvP_B_minus {
                    if AvB_minus <= AvP_B_minus {
                        P_B_minus_index -= 1;
                        let AvP_B_minus = V.pop().unwrap();
                        let P_B_minus_w = W.pop().unwrap();
                        //println!(
                        //        "AvB-:{AvB_minus}*{B_minus_w}, AvP(B-):{AvP_B_minus}*{P_B_minus_w}, P(B-)[{P_B_minus_index}]"
                        //    );

                        // New Av(B-)
                        AvB_minus = (AvB_minus * B_minus_w + AvP_B_minus * P_B_minus_w)
                            / (B_minus_w + P_B_minus_w);
                        B_minus_w += P_B_minus_w;

                        // New B-
                        let P_B_minus_start = J_index[P_B_minus_index];
                        J_index[P_B_minus_start] = B_minus_index;
                        J_index[B_minus_index] = P_B_minus_index;
                        //println!(
                        //    "P(B-)[{P_B_minus_index}], V:{:?} W:{:?}, J:{:?}",
                        //    V, W, J_index
                        //);
                    }
                }

                V.push(AvB_minus);
                W.push(B_minus_w);
                i = B_minus_index + 1;
                j = 1;
                //println!("i:{i} j:{j}, V:{:?} W:{:?}, J:{:?}", V, W, J_index);
            }
        }
        //println!("AA] i:{i}, j:{j} {:?}", J_index);

        // Last block average
        let (AvB_minus, _) = waverage(&ys, weights, i, J_index[i], &index);
        V.push(AvB_minus);

        (V, J_index)
    }
}

/// An implementation of PVA algorithm from Best (1990)
/// for solving IRC problem
impl IR for PVA {
    fn evaluate<F, D>(
        &self,
        ys: &ArrayBase<D, Ix1>,
        weights: Option<&[f32]>,
        index: &Vec<usize>,
    ) -> (Vec<F>, Vec<usize>)
    where
        F: Float,
        D: Data<Elem = F>,
    {
        let n = ys.len();
        let mut V = Vec::<F>::new();
        let mut W = Vec::<F>::new();
        let mut J_index: Vec<usize> = (0..n).collect();
        let mut i = 0;
        let (mut AvB_zero, mut W_B_zero) = waverage(&ys, weights, i, i, &index);
        //println!("Y:{:?}", ys);
        //println!("I:{:?}", index);
        //println!("y:{:?}", ys.select(Axis(0), &index));
        while i < n {
            // Step 1
            let j = J_index[i];
            let k = j + 1;
            if k == n {
                break;
            }
            let l = J_index[k];
            let (AvB_plus, W_B_plus) = waverage(&ys, weights, k, l, &index);
            //println!("i:{i} j:{j}, k:{k}, l:{l}, Av(B₀):{AvB_zero}[{W_B_zero}], Av(B₊):{AvB_plus}[{W_B_plus}]");
            if AvB_zero <= AvB_plus {
                V.push(AvB_zero);
                W.push(W_B_zero);
                AvB_zero = AvB_plus;
                W_B_zero = W_B_plus;
                i = k;
                //println!(
                //    "i:{i} Av(B₀):{AvB_zero}[{W_B_zero}], Av(B₊):{AvB_plus}[{W_B_plus}], {:?}",
                //    J_index
                //);
            } else {
                // Step 2
                J_index[i] = l;
                J_index[l] = i;
                AvB_zero = AvB_zero * W_B_zero + AvB_plus * W_B_plus;
                W_B_zero += W_B_plus;
                AvB_zero /= W_B_zero;
                //println!("i:{i} j:{j}, k:{k}, l:{l}, Av(B₀):{AvB_zero}[{W_B_zero}], {:?}", J_index);
                // Step 2.1
                let mut AvB_minus = *V.last().unwrap_or(&F::neg_infinity());
                while V.len() > 0 && AvB_zero <= AvB_minus {
                    AvB_minus = V.pop().unwrap();
                    let W_B_minus = W.pop().unwrap();
                    i = J_index[J_index[l] - 1];
                    J_index[l] = i;
                    J_index[i] = l;
                    AvB_zero = AvB_zero * W_B_zero + AvB_minus * W_B_minus;
                    W_B_zero += W_B_minus;
                    AvB_zero /= W_B_zero;
                    //println!("i:{i} j:{j}, Av(B₀):{AvB_zero}[{W_B_zero}], Av(B₋):{AvB_minus}[{W_B_minus}], {:?}", J_index);
                }
            }
            //println!("i:{i} V:{:?} W:{:?}, J:{:?}", V, W, J_index);
        }

        // Last block average
        let (AvB_minus, _) = waverage(&ys, weights, i, J_index[i], &index);
        V.push(AvB_minus);

        (V, J_index)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
/// An isotonic regression model.
///
/// IsotonicRegression solves an isotonic regression problem using the pool
/// adjacent violators algorithm.
///
/// /// ## Examples
///
/// Here's an example on how to train an isotonic regression model on
/// the first feature from the `diabetes` dataset.
/// ```rust
/// use linfa::{traits::Fit, traits::Predict, Dataset};
/// use linfa_linear::IsotonicRegression;
/// use linfa::prelude::SingleTargetRegression;
///
/// let diabetes = linfa_datasets::diabetes();
/// let dataset = diabetes.feature_iter().next().unwrap(); // get first 1D feature
/// let model = IsotonicRegression::default().fit(&dataset).unwrap();
/// let pred = model.predict(&dataset);
/// let r2 = pred.r2(&dataset).unwrap();
/// println!("r2 from prediction: {}", r2);
/// ```
pub struct IsotonicRegression<A: IR = PVA> {
    algo: A,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
/// A fitted isotonic regression model which can be used for making predictions.
pub struct FittedIsotonicRegression<F> {
    regressor: Array1<F>,
    response: Array1<F>,
}

/// Configure and fit a isotonic regression model
impl<A: IR + Default> IsotonicRegression<A> {
    /// Create a default isotonic regression model.
    pub fn new() -> IsotonicRegression<A> {
        IsotonicRegression { algo: A::default() }
    }
}

impl Default for IsotonicRegression {
    fn default() -> Self {
        <IsotonicRegression>::new()
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = F>, A: IR>
    Fit<ArrayBase<D, Ix2>, T, LinearError<F>> for IsotonicRegression<A>
{
    type Object = FittedIsotonicRegression<F>;

    /// Fit an isotonic regression model given a feature matrix `X` and a target
    /// variable `y`.
    ///
    /// The feature matrix `X` must have shape `(n_samples, 1)`
    ///
    /// The target variable `y` must have shape `(n_samples)`
    ///
    /// Returns a `FittedIsotonicRegression` object which contains the fitted
    /// parameters and can be used to `predict` values of the target variable
    /// for new feature values.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object, F> {
        let X = dataset.records();
        let (n, dim) = X.dim();
        let y = dataset.as_single_targets();

        // Check the input dimension
        assert_eq!(dim, 1, "The input dimension must be 1.");

        // Check that our inputs have compatible shapes
        assert_eq!(y.dim(), n);

        // use correlation for determining relationship between x & y
        let x = X.column(0);
        let rho = DatasetBase::from(stack![Axis(1), x, y]).pearson_correlation();
        let increasing = rho.get_coeffs()[0] >= F::zero();

        // sort data
        let mut indices = argsort_by(&x, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
        if !increasing {
            indices.reverse();
        };

        // Construct response value using particular algorithm
        let (V, J_index) = self.algo.evaluate(&y, dataset.weights(), &indices);
        let response = Array1::from_vec(V.clone());

        // Construct regressor array
        let mut W = Vec::<F>::new();
        let mut i = 0;
        while i < n {
            let j = J_index[i];
            let x = X
                .slice(s![i..=j, -1])
                .into_iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater))
                .unwrap();
            W.push(*x);
            i = j + 1
        }
        let regressor = Array1::from_vec(W.clone());

        Ok(FittedIsotonicRegression {
            regressor,
            response,
        })
    }
}

fn waverage<F, D>(
    vs: &ArrayBase<D, Ix1>,
    ws: Option<&[f32]>,
    start: usize,
    end: usize,
    index: &Vec<usize>,
) -> (F, F)
where
    F: Float,
    D: Data<Elem = F>,
{
    let mut wsum = F::zero();
    let mut avg = F::zero();
    for k in start..=end {
        let kk = index[k];
        let w = if ws.is_none() {
            F::one()
        } else {
            F::cast(ws.unwrap()[kk])
        };
        wsum += w;
        avg += vs[kk] * w;
    }
    avg /= wsum;
    (avg, wsum)
}

fn argsort_by<S, F>(arr: &ArrayBase<S, Ix1>, mut compare: F) -> Vec<usize>
where
    S: Data,
    F: FnMut(&S::Elem, &S::Elem) -> Ordering,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_unstable_by(move |&i, &j| compare(&arr[i], &arr[j]));
    indices
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<F>>
    for FittedIsotonicRegression<F>
{
    /// Given an input matrix `X`, with shape `(n_samples, 1)`,
    /// `predict` returns the target variable according to linear model
    /// learned from the training data distribution.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<F>) {
        let (n_samples, dim) = x.dim();

        // Check the input dimension
        assert_eq!(dim, 1, "The input dimension must be 1.");

        // Check that our inputs have compatible shapes
        assert_eq!(
            n_samples,
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let regressor = &self.regressor;
        let n = regressor.len();
        let x_min = regressor[0];
        let x_max = regressor[n - 1];

        let response = &self.response;
        let y_min = response[0];
        let y_max = response[n - 1];

        // calculate a piecewise linear approximation of response
        for (i, row) in x.rows().into_iter().enumerate() {
            let val = row[0];
            if val >= x_max {
                y[i] = y_max;
            } else if val <= x_min {
                y[i] = y_min;
            } else {
                let res = regressor.into_iter().position(|x| x >= &val);
                if res.is_some() {
                    let j = res.unwrap();
                    if val <= regressor[j] && j < n {
                        let x_scale = (val - regressor[j - 1]) / (regressor[j] - regressor[j - 1]);
                        y[i] = response[j - 1] + x_scale * (response[j] - response[j - 1]);
                    } else {
                        y[i] = y_min;
                    }
                }
            }
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        Array1::zeros(x.nrows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::{traits::Predict, Dataset};
    use ndarray::array;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<FittedIsotonicRegression<f64>>();
        has_autotraits::<IsotonicRegression<AlgorithmA>>();
        has_autotraits::<LinearError<f64>>();
    }

    #[test]
    #[should_panic]
    fn dimension_mismatch() {
        let reg = <IsotonicRegression>::new();
        let dataset = Dataset::new(array![[3.3f64, 0.], [3.3, 0.]], array![4., 5.]);
        let _res = reg.fit(&dataset);
        ()
    }

    #[test]
    #[should_panic]
    fn length_mismatch() {
        let reg = IsotonicRegression::default();
        let dataset = Dataset::new(array![[3.3f64, 0.], [3.3, 0.]], array![4., 5., 6.]);
        let _res = reg.fit(&dataset);
        ()
    }

    fn best_example1<R: IR>(reg: &IsotonicRegression<R>) {
        let (X, y, regr, resp, yy, V, w) = (
            array![[3.3f64], [3.3], [3.3], [6.], [7.5], [7.5]], // X
            array![4., 5., 1., 6., 8., 7.0],                    // y
            array![3.3, 6., 7.5],                               // regressor
            array![10.0 / 3.0, 6., 7.5],                        // response
            array![10. / 3., 10. / 3., 10. / 3., 6., 7.5, 7.5], // predict X
            array![[2.0f64], [5.], [7.], [9.]],                 // newX
            array![10. / 3., 5.01234567901234, 7., 7.5],        // predict newX
        );

        let dataset = Dataset::new(X, y);

        let model = reg.fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.regressor, &regr, epsilon = 1e-12);
        assert_abs_diff_eq!(model.response, &resp, epsilon = 1e-12);

        let result = model.predict(dataset.records());
        assert_abs_diff_eq!(result, &yy, epsilon = 1e-12);

        let result = model.predict(&V);
        assert_abs_diff_eq!(result, &w, epsilon = 1e-12);
    }

    fn best_example1_decr<R: IR>(reg: &IsotonicRegression<R>) {
        let (X, y, regr, resp, yy, V, w) = (
            array![[7.5], [7.5], [6.], [3.3], [3.3], [3.3]], // X
            array![4., 5., 1., 6., 8., 7.0],                 // y
            array![3.3, 6., 7.5],                            // regressor
            array![10.0 / 3.0, 6., 7.5],                     // response
            array![10. / 3., 10. / 3., 10. / 3., 6., 7.5, 7.5], // predict X
            array![[2.0f64], [5.], [7.], [9.]],              // newX
            array![10. / 3., 5.01234567901234, 7., 7.5],     // predict newX
        );

        let dataset = Dataset::new(X, y);

        let model = reg.fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.regressor, &regr, epsilon = 1e-12);
        assert_abs_diff_eq!(model.response, &resp, epsilon = 1e-12);

        let result = model.predict(dataset.records());
        assert_abs_diff_eq!(result, &yy, epsilon = 1e-12);

        let result = model.predict(&V);
        assert_abs_diff_eq!(result, &w, epsilon = 1e-12);
    }

    fn example2_incr<R: IR>(reg: &IsotonicRegression<R>) {
        let is_pva = std::any::type_name::<R>().ends_with("PVA");
        let (X, y, regr, resp, yy) = (
            array![[1.0f64], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]],
            array![1., 2., 6., 2., 1., 2., 8., 2., 1.0],
            if is_pva {
                array![1., 2., 6., 9.]
            } else {
                array![6., 9.]
            },
            if is_pva {
                array![1., 2., 2.75, 11. / 3.]
            } else {
                array![7. / 3., 11. / 3.]
            },
            if is_pva {
                array![
                    1.,
                    2.,
                    2.1875,
                    2.375,
                    2.5625,
                    2.75,
                    55. / 18.,
                    121. / 36.,
                    11. / 3.
                ]
            } else {
                let v1 = 7. / 3.;
                array![v1, v1, v1, v1, v1, v1, 25. / 9., 29. / 9., 11. / 3.]
            },
        );

        let dataset = Dataset::new(X, y);

        let model = reg.fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.regressor, &regr, epsilon = 1e-12);
        assert_abs_diff_eq!(model.response, &resp, epsilon = 1e-12);

        let result = model.predict(dataset.records());
        assert_abs_diff_eq!(result, &yy, epsilon = 1e-12);
    }

    fn example2_decr<R: IR>(reg: &IsotonicRegression<R>) {
        let is_pva = std::any::type_name::<R>().ends_with("PVA");
        let (X, y, regr, resp, yy) = (
            array![[1.0f64], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]],
            array![1., 2., 6., 2., 1., 2., 8., 2., 1.0],
            if is_pva {
                array![1., 2., 6., 9.]
            } else {
                array![6., 9.]
            },
            if is_pva {
                array![1., 2., 2.75, 11. / 3.]
            } else {
                array![7. / 3., 11. / 3.]
            },
            if is_pva {
                array![
                    1.,
                    2.,
                    2.1875,
                    2.375,
                    2.5625,
                    2.75,
                    55. / 18.,
                    121. / 36.,
                    11. / 3.
                ]
            } else {
                let v1 = 7. / 3.;
                array![v1, v1, v1, v1, v1, v1, 25. / 9., 29. / 9., 11. / 3.]
            },
        );

        let dataset = Dataset::new(X, y);

        let model = reg.fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.regressor, &regr, epsilon = 1e-12);
        assert_abs_diff_eq!(model.response, &resp, epsilon = 1e-12);

        let result = model.predict(dataset.records());
        assert_abs_diff_eq!(result, &yy, epsilon = 1e-12);
    }

    #[test]
    fn best_example1_incr_pva() {
        let reg = IsotonicRegression::<PVA>::new();
        best_example1(&reg);
    }

    #[test]
    fn best_example1_incr_algoa() {
        let reg = IsotonicRegression::<AlgorithmA>::new();
        best_example1(&reg);
    }

    #[test]
    fn best_example1_decr_pva() {
        let reg = IsotonicRegression::<PVA>::new();
        best_example1_decr(&reg);
    }

    #[test]
    #[ignore]
    fn best_example1_decr_algoa() {
        let reg = IsotonicRegression::<AlgorithmA>::new();
        best_example1_decr(&reg);
    }

    #[test]
    fn example2_incr_pva() {
        let reg = IsotonicRegression::default();
        example2_incr(&reg);
    }

    #[test]
    fn example2_incr_algoa() {
        let reg = IsotonicRegression::<AlgorithmA>::new();
        example2_incr(&reg);
    }

    #[test]
    #[ignore]
    fn example2_decr_pva() {
        let reg = IsotonicRegression::default();
        example2_decr(&reg);
    }

    #[test]
    #[ignore]
    fn example2_decr_algoa() {
        let reg = IsotonicRegression::<AlgorithmA>::new();
        example2_decr(&reg);
    }
}
