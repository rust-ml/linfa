use linfa::prelude::Transformer;
use linfa::{
    composing::platt_scaling::{platt_newton_method, platt_predict, PlattParams},
    dataset::{AsTargets, CountedTargets, DatasetBase, Pr},
    traits::Fit,
    traits::{Predict, PredictRef},
};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Data, Ix1, Ix2};
use std::cmp::Ordering;

use super::error::{Result, SvmResult};
use super::permutable_kernel::{PermutableKernel, PermutableKernelOneClass};
use super::solver_smo::SolverState;
use super::SolverParams;
use super::{Float, Svm, SvmParams};
use linfa_kernel::Kernel;

fn calibrate_with_platt<F: Float, D: Data<Elem = F>, T: AsTargets<Elem = bool>>(
    mut obj: Svm<F, F>,
    params: &PlattParams<F, ()>,
    dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
) -> Result<Svm<F, Pr>> {
    let pred = dataset
        .records()
        .outer_iter()
        .map(|x| obj.weighted_sum(&x) - obj.rho)
        .collect::<Array1<_>>();

    let (a, b) = platt_newton_method(pred.view(), dataset.try_single_target()?, params)?;
    obj.probability_coeffs = Some((a, b));

    Ok(obj.with_phantom())
}

/// Support Vector Classification with C-penalizing parameter
///
/// This methods solves a binary SVC problem with a penalizing parameter C between (0, inf). The
/// dual problem has the form
/// ```ignore
/// min_a 1/2*a^tQ a - e^T a s.t. y^t = 0, 0 <= a_i <= C_i
/// ```
/// with `Q_ij = y_i y_j K(x_i, x_j)` the kernel matrix.
///
/// # Parameters
///
/// * `params` - Solver parameters (threshold etc.)
/// * `kernel` - the kernel matrix `Q`
/// * `targets` - the ground truth targets `y_i`
/// * `cpos` - C for positive targets
/// * `cneg` - C for negative targets
pub fn fit_c<F: Float>(
    params: SolverParams<F>,
    dataset: ArrayView2<F>,
    kernel: Kernel<F>,
    targets: &[bool],
    cpos: F,
    cneg: F,
) -> Svm<F, F> {
    let bounds = targets
        .iter()
        .map(|x| if *x { cpos } else { cneg })
        .collect::<Vec<_>>();

    let kernel = PermutableKernel::new(kernel, targets.to_vec());

    let solver = SolverState::new(
        vec![F::zero(); targets.len()],
        vec![-F::one(); targets.len()],
        targets.to_vec(),
        dataset,
        kernel,
        bounds,
        params,
        false,
    );

    let mut res = solver.solve();

    res.alpha = res
        .alpha
        .into_iter()
        .zip(targets.iter())
        .map(|(a, b)| if *b { a } else { -a })
        .collect();

    res
}

/// Support Vector Classification with Nu-penalizing term
///
/// This methods solves a binary SVC problem with a penalizing parameter nu between (0, 1). The
/// dual problem has the form
/// ```ignore
/// min_a 1/2*a^tQ a s.t. y^t a = 0, 0 <= a_i <= 1/l, e^t a > nu
/// ```
/// with `Q_ij = y_i y_j K(x_i, x_j)` the kernel matrix.
///
/// # Parameters
///
/// * `params` - Solver parameters (threshold etc.)
/// * `kernel` - the kernel matrix `Q`
/// * `targets` - the ground truth targets `y_i`
/// * `nu` - Nu penalizing term
pub fn fit_nu<F: Float>(
    params: SolverParams<F>,
    dataset: ArrayView2<F>,
    kernel: Kernel<F>,
    targets: &[bool],
    nu: F,
) -> Svm<F, F> {
    let mut sum_pos = nu * F::cast(targets.len()) / F::cast(2.0);
    let mut sum_neg = nu * F::cast(targets.len()) / F::cast(2.0);
    let init_alpha = targets
        .iter()
        .map(|x| {
            if *x {
                let val = F::min(F::one(), sum_pos);
                sum_pos -= val;
                val
            } else {
                let val = F::min(F::one(), sum_neg);
                sum_neg -= val;
                val
            }
        })
        .collect::<Vec<_>>();

    let kernel = PermutableKernel::new(kernel, targets.to_vec());

    let solver = SolverState::new(
        init_alpha,
        vec![F::zero(); targets.len()],
        targets.to_vec(),
        dataset,
        kernel,
        vec![F::one(); targets.len()],
        params,
        true,
    );

    let mut res = solver.solve();

    let r = res.r.unwrap();

    res.alpha = res
        .alpha
        .into_iter()
        .zip(targets.iter())
        .map(|(a, b)| if *b { a } else { -a })
        .map(|x| x / r)
        .collect();
    res.rho /= r;
    res.obj /= r * r;

    res
}

/// Support Vector Classification for one-class problems
///
/// This methods solves a binary SVC, when there are no targets available. This can, for example be
/// useful, when outliers should be rejected.
///
/// # Parameters
///
/// * `params` - Solver parameters (threshold etc.)
/// * `kernel` - the kernel matrix `Q`
/// * `nu` - Nu penalizing term
pub fn fit_one_class<F: Float + num_traits::ToPrimitive>(
    params: SolverParams<F>,
    dataset: ArrayView2<F>,
    kernel: Kernel<F>,
    nu: F,
) -> Svm<F, F> {
    let size = kernel.size();
    let n = (nu * F::cast(size)).to_usize().unwrap();

    let init_alpha = (0..size)
        .map(|x| match x.cmp(&n) {
            Ordering::Less => F::one(),
            Ordering::Greater => F::zero(),
            Ordering::Equal => nu * F::cast(size) - F::cast(x),
        })
        .collect::<Vec<_>>();

    let kernel = PermutableKernelOneClass::new(kernel);

    let solver = SolverState::new(
        init_alpha,
        vec![F::zero(); size],
        vec![true; size],
        dataset,
        kernel,
        vec![F::one(); size],
        params,
        false,
    );

    let res = solver.solve();

    res
}

/// Fit binary classification problem
///
/// For a given dataset with kernel matrix as records and two class problem as targets this fits
/// a optimal hyperplane to the problem and returns the solution as a model. The model predicts
/// probabilities for whether a sample belongs to the first or second class.
macro_rules! impl_classification {
    ($records:ty, $targets:ty) => {
        impl<F: Float> Fit<$records, $targets, SvmResult> for SvmParams<F, Pr> {
            type Object = Svm<F, Pr>;

            fn fit(&self, dataset: &DatasetBase<$records, $targets>) -> Result<Self::Object> {
                let kernel = self.kernel.transform(dataset.records());
                let target = dataset.try_single_target()?;
                let target = target.as_slice().unwrap();

                let ret: Self::Object = match (self.c, self.nu) {
                    (Some((c_p, c_n)), _) => fit_c(
                        self.solver_params.clone(),
                        dataset.records().view(),
                        kernel,
                        target,
                        c_p,
                        c_n,
                    ),
                    (None, Some((nu, _))) => fit_nu(
                        self.solver_params.clone(),
                        dataset.records().view(),
                        kernel,
                        target,
                        nu,
                    ),
                    _ => panic!("Set either C value or Nu value"),
                };

                calibrate_with_platt(ret, &self.platt, dataset)
            }
        }

        impl<'a, F: Float> Fit<'a, $records, $targets> for SvmParams<F, bool> {
            type Object = Result<Svm<F, bool>>;

            fn fit(&self, dataset: &DatasetBase<$records, $targets>) -> Self::Object {
                let kernel = self.kernel.transform(dataset.records());
                let target = dataset.try_single_target()?;
                let target = target.as_slice().unwrap();

                let ret = match (self.c, self.nu) {
                    (Some((c_p, c_n)), _) => fit_c(
                        self.solver_params.clone(),
                        dataset.records().view(),
                        kernel,
                        target,
                        c_p,
                        c_n,
                    ),
                    (None, Some((nu, _))) => fit_nu(
                        self.solver_params.clone(),
                        dataset.records().view(),
                        kernel,
                        target,
                        nu,
                    ),
                    _ => panic!("Set either C value or Nu value"),
                };

                Ok(ret.with_phantom())
            }
        }
    };
}

impl_classification!(Array2<F>, Array2<bool>);
impl_classification!(ArrayView2<'_, F>, ArrayView2<'_, bool>);
impl_classification!(Array2<F>, CountedTargets<bool, Array2<bool>>);
impl_classification!(ArrayView2<'_, F>, CountedTargets<bool, ArrayView2<'_, bool>>);

/// Fit one-class problem
///
/// This fits a SVM model to a dataset with only positive samples and uses the one-class
/// implementation of SVM.
macro_rules! impl_oneclass {
    ($records:ty, $targets:ty) => {
        impl<F: Float> Fit<$records, $targets, SvmResult> for SvmParams<F, Pr> {
            type Object = Svm<F, Pr>;

            fn fit(&self, dataset: &DatasetBase<$records, $targets>) -> Result<Self::Object> {
                let kernel = self.kernel.transform(dataset.records());
                let records = dataset.records().view();

                let ret = match self.nu {
                    Some((nu, _)) => fit_one_class(self.solver_params.clone(), records, kernel, nu),
                    None => panic!("One class needs Nu value"),
                };

                Ok(ret.with_phantom())
            }
        }
    };
}

impl_oneclass!(Array2<F>, Array2<()>);
impl_oneclass!(ArrayView2<'_, F>, ArrayView2<'_, ()>);
impl_oneclass!(Array2<F>, CountedTargets<(), Array2<()>>);
impl_oneclass!(Array2<F>, CountedTargets<(), ArrayView2<'_, ()>>);

/// Predict a probability with a feature vector
impl<F: Float, D: Data<Elem = F>> Predict<ArrayBase<D, Ix1>, Pr> for Svm<F, Pr> {
    fn predict(&self, data: ArrayBase<D, Ix1>) -> Pr {
        let val = self.weighted_sum(&data) - self.rho;
        let (a, b) = self.probability_coeffs.clone().unwrap();

        platt_predict(val, a, b)
    }
}

/// Predict a probability with a feature vector
impl<'a, F: Float, D: Data<Elem = F>> Predict<ArrayBase<D, Ix1>, bool> for Svm<F, bool> {
    fn predict(&self, data: ArrayBase<D, Ix1>) -> bool {
        let val = self.weighted_sum(&data) - self.rho;

        val >= F::zero()
    }
}

/// Predict a probability with a feature vector
/*impl<'a, F: Float> Predict<ArrayView1<'a, F>, Pr> for Svm<F, Pr> {
    fn predict(&self, data: ArrayView1<'a, F>) -> Pr {
        let val = self.weighted_sum(&data) - self.rho;
        let (a, b) = self.probability_coeffs.clone().unwrap();

        platt_predict(val, a, b)
    }
}

/// Predict a probability with a feature vector
impl<F: Float> Predict<Array1<F>, bool> for Svm<F, bool> {
    fn predict(&self, data: Array1<F>) -> bool {
        let val = self.weighted_sum(&data) - self.rho;

        val >= F::zero()
    }
}*/

/// Classify observations
///
/// This function takes a number of features and predicts target probabilities that they belong to
/// the positive class.
impl<F: Float, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix2>, Array1<Pr>> for Svm<F, Pr> {
    fn predict_ref<'a>(&'a self, data: &ArrayBase<D, Ix2>) -> Array1<Pr> {
        let (a, b) = self.probability_coeffs.clone().unwrap();

        data.outer_iter()
            .map(|data| {
                let val = self.weighted_sum(&data) - self.rho;
                platt_predict(val, a, b)
            })
            .collect()
    }
}

/// Classify observations
///
/// This function takes a number of features and predicts target probabilities that they belong to
/// the positive class.
impl<F: Float, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix2>, Array1<bool>> for Svm<F, bool> {
    fn predict_ref<'a>(&'a self, data: &ArrayBase<D, Ix2>) -> Array1<bool> {
        data.outer_iter()
            .map(|data| {
                let val = self.weighted_sum(&data) - self.rho;

                val >= F::zero()
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::Svm;
    use crate::error::Result;
    use linfa::dataset::{Dataset, DatasetBase};
    use linfa::prelude::ToConfusionMatrix;
    use linfa::traits::{Fit, Predict};

    use ndarray::{Array, Array1, Array2, Axis};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_isaac::Isaac64Rng;

    pub fn generate_convoluted_rings(n_points: usize) -> Array2<f64> {
        let mut out = Array::random((n_points * 2, 2), Uniform::new(0f64, 1.));
        for (i, mut elm) in out.outer_iter_mut().enumerate() {
            // generate convoluted rings with 1/10th noise
            let phi = 6.28 * elm[1];
            let eps = elm[0] / 10.0;

            if i < n_points {
                elm[0] = 1.0 * phi.cos() + eps;
                elm[1] = 1.0 * phi.sin() + eps;
            } else {
                elm[0] = 5.0 * phi.cos() + eps;
                elm[1] = 5.0 * phi.sin() + eps;
            }
        }

        out
    }

    #[test]
    fn test_linear_classification() -> Result<()> {
        let entries: Array2<f64> = ndarray::concatenate(
            Axis(0),
            &[
                Array::random((10, 2), Uniform::new(-1., -0.5)).view(),
                Array::random((10, 2), Uniform::new(0.5, 1.)).view(),
            ],
        )
        .unwrap();
        let targets = (0..20).map(|x| x < 10).collect::<Array1<_>>();
        let dataset = Dataset::new(entries.clone(), targets);

        // train model with positive and negative weight
        let model = Svm::<_, bool>::params()
            .pos_neg_weights(1.0, 1.0)
            .linear_kernel()
            .fit(&dataset)?;

        let y_est = model.predict(&dataset);

        let cm = y_est.confusion_matrix(&dataset)?;
        assert_eq!(cm.accuracy(), 1.0);

        // train model with Nu parameter
        let model = Svm::<_, bool>::params()
            .nu_weight(0.05)
            .linear_kernel()
            .fit(&dataset)?;

        let valid = model.predict(&dataset);

        let cm = valid.confusion_matrix(&dataset)?;
        assert_eq!(cm.accuracy(), 1.0);

        Ok(())
    }

    #[test]
    fn test_polynomial_classification() -> Result<()> {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        // construct parabolica and classify middle area as positive and borders as negative
        let records = Array::random_using((40, 1), Uniform::new(-2f64, 2.), &mut rng);
        let targets = records.map_axis(Axis(1), |x| x[0] * x[0] < 0.5);
        let dataset = Dataset::new(records.clone(), targets);

        // train model with positive and negative weight
        let model = Svm::<_, bool>::params()
            .pos_neg_weights(1.0, 1.0)
            .polynomial_kernel(0.0, 2.0)
            .fit(&dataset)?;

        //println!("{:?}", model.predict(DatasetBase::from(records.clone())).targets());

        let valid = model.predict(&dataset);

        let cm = valid.confusion_matrix(&dataset)?;
        assert!(cm.accuracy() > 0.9);

        Ok(())
    }

    #[test]
    fn test_convoluted_rings_classification() -> Result<()> {
        let records = generate_convoluted_rings(10);
        let targets = (0..20).map(|x| x < 10).collect::<Array1<_>>();
        let dataset = (records.view(), targets.view()).into();

        // train model with positive and negative weight
        let model = Svm::<_, bool>::params()
            .pos_neg_weights(1.0, 1.0)
            .gaussian_kernel(50.0)
            .fit(&dataset)?;

        let y_est = model.predict(&dataset);

        let cm = y_est.confusion_matrix(&dataset)?;
        assert!(cm.accuracy() > 0.9);

        // train model with Nu parameter
        let model = Svm::<_, bool>::params()
            .nu_weight(0.01)
            .gaussian_kernel(50.0)
            .fit(&dataset)?;

        let y_est = model.predict(&dataset);

        let cm = y_est.confusion_matrix(&dataset)?;
        assert!(cm.accuracy() > 0.9);

        Ok(())
    }

    #[test]
    fn test_iris_crossvalidation() {
        let params = Svm::<_, bool>::params()
            .pos_neg_weights(50000., 5000.)
            .gaussian_kernel(40.0);

        // perform cross-validation with the MCC
        let acc_runs = linfa_datasets::winequality()
            .map_targets(|x| *x > 6)
            .iter_fold(1, |v| params.fit(&v).unwrap())
            .map(|(model, valid)| {
                let cm = model.predict(&valid).confusion_matrix(&valid).unwrap();

                cm.accuracy()
            })
            .collect::<Array1<_>>();

        assert!(acc_runs[0] > 0.85);
    }

    #[test]
    fn test_reject_classification() -> Result<()> {
        // generate two clusters with 100 samples each
        let entries = Array::random((100, 2), Uniform::new(-4., 4.));
        let dataset = Dataset::from(entries);

        // train model with positive and negative weight
        let model = Svm::params()
            .nu_weight(1.0)
            .gaussian_kernel(100.0)
            .fit(&dataset)?;

        let valid = DatasetBase::from(Array::random((100, 2), Uniform::new(-10., 10f32)));
        let valid = model.predict(valid);

        // count the number of correctly rejected samples
        let mut rejected = 0;
        let mut total = 0;
        for (pred, pos) in valid.targets().iter().zip(valid.records.outer_iter()) {
            let distance = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
            if distance >= 5.0 {
                if !pred {
                    rejected += 1;
                }
                total += 1;
            }
        }

        // at least 95% should be correctly rejected
        assert!((rejected as f32) / (total as f32) > 0.95);

        Ok(())
    }
}
