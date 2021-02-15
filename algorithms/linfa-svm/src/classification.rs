use linfa::prelude::Transformer;
use linfa::{dataset::DatasetBase, dataset::Pr, dataset::Targets, traits::Fit, traits::Predict};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};
use std::cmp::Ordering;

use super::permutable_kernel::{PermutableKernel, PermutableKernelOneClass};
use super::solver_smo::SolverState;
use super::SolverParams;
use super::{Float, Svm, SvmParams};
use linfa_kernel::Kernel;

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
) -> Svm<F, Pr> {
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

    res.with_phantom()
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
) -> Svm<F, Pr> {
    let mut sum_pos = nu * F::from(targets.len()).unwrap() / F::from(2.0).unwrap();
    let mut sum_neg = nu * F::from(targets.len()).unwrap() / F::from(2.0).unwrap();
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

    res.with_phantom()
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
) -> Svm<F, Pr> {
    let size = kernel.size();
    let n = (nu * F::from(size).unwrap()).to_usize().unwrap();

    let init_alpha = (0..size)
        .map(|x| match x.cmp(&n) {
            Ordering::Less => F::one(),
            Ordering::Greater => F::zero(),
            Ordering::Equal => nu * F::from(size).unwrap() - F::from(x).unwrap(),
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

    res.with_phantom()
}

/// Fit binary classification problem
///
/// For a given dataset with kernel matrix as records and two class problem as targets this fits
/// a optimal hyperplane to the problem and returns the solution as a model. The model predicts
/// probabilities for whether a sample belongs to the first or second class.
impl<'a, F: Float> Fit<'a, Array2<F>, &Array1<bool>> for SvmParams<F, Pr> {
    type Object = Svm<F, Pr>;

    fn fit(&self, dataset: &DatasetBase<Array2<F>, &Array1<bool>>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match (self.c, self.nu) {
            (Some((c_p, c_n)), _) => fit_c(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice(),
                c_p,
                c_n,
            ),
            (None, Some((nu, _))) => fit_nu(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice(),
                nu,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, Array2<F>, Array1<bool>> for SvmParams<F, Pr> {
    type Object = Svm<F, Pr>;

    fn fit(&self, dataset: &DatasetBase<Array2<F>, Array1<bool>>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match (self.c, self.nu) {
            (Some((c_p, c_n)), _) => fit_c(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                c_p,
                c_n,
            ),
            (None, Some((nu, _))) => fit_nu(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                nu,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, Array2<F>, ArrayView1<'a, bool>> for SvmParams<F, Pr> {
    type Object = Svm<F, Pr>;

    fn fit(&self, dataset: &DatasetBase<Array2<F>, ArrayView1<'a, bool>>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match (self.c, self.nu) {
            (Some((c_p, c_n)), _) => fit_c(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                c_p,
                c_n,
            ),
            (None, Some((nu, _))) => fit_nu(
                self.solver_params.clone(),
                dataset.records().view(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                nu,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, ArrayView2<'a, F>, ArrayView1<'a, bool>> for SvmParams<F, Pr> {
    type Object = Svm<F, Pr>;

    fn fit(&self, dataset: &DatasetBase<ArrayView2<'a, F>, ArrayView1<'a, bool>>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match (self.c, self.nu) {
            (Some((c_p, c_n)), _) => fit_c(
                self.solver_params.clone(),
                *dataset.records(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                c_p,
                c_n,
            ),
            (None, Some((nu, _))) => fit_nu(
                self.solver_params.clone(),
                *dataset.records(),
                kernel,
                dataset.targets().as_slice().unwrap(),
                nu,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, ArrayView2<'a, F>, &[bool]> for SvmParams<F, Pr> {
    type Object = Svm<F, Pr>;

    fn fit(&self, dataset: &DatasetBase<ArrayView2<'a, F>, &[bool]>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match (self.c, self.nu) {
            (Some((c_p, c_n)), _) => fit_c(
                self.solver_params.clone(),
                *dataset.records(),
                kernel,
                dataset.targets(),
                c_p,
                c_n,
            ),
            (None, Some((nu, _))) => fit_nu(
                self.solver_params.clone(),
                *dataset.records(),
                kernel,
                dataset.targets(),
                nu,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

/// Fit one-class problem
///
/// This fits a SVM model to a dataset with only positive samples and uses the one-class
/// implementation of SVM.
impl<'a, F: Float> Fit<'a, ArrayView2<'a, F>, &()> for SvmParams<F, Pr> {
    type Object = Svm<F, Pr>;

    fn fit(&self, dataset: &DatasetBase<ArrayView2<'a, F>, &()>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match self.nu {
            Some((nu, _)) => {
                fit_one_class(self.solver_params.clone(), *dataset.records(), kernel, nu)
            }
            None => panic!("One class needs Nu value"),
        }
    }
}

/// Fit one-class problem
///
/// This fits a SVM model to a dataset with only positive samples and uses the one-class
/// implementation of SVM.
impl<'a, F: Float> Fit<'a, ArrayView2<'a, F>, ()> for SvmParams<F, Pr> {
    type Object = Svm<F, Pr>;

    fn fit(&self, dataset: &DatasetBase<ArrayView2<'a, F>, ()>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match self.nu {
            Some((nu, _)) => {
                fit_one_class(self.solver_params.clone(), *dataset.records(), kernel, nu)
            }
            None => panic!("One class needs Nu value"),
        }
    }
}

/// Fit one-class problem
///
/// This fits a SVM model to a dataset with only positive samples and uses the one-class
/// implementation of SVM.
impl<'a, F: Float> Fit<'a, ArrayView2<'a, F>, &[()]> for SvmParams<F, Pr> {
    type Object = Svm<F, Pr>;

    fn fit(&self, dataset: &DatasetBase<ArrayView2<'a, F>, &[()]>) -> Self::Object {
        let kernel = self.kernel.transform(dataset.records());
        match self.nu {
            Some((nu, _)) => {
                fit_one_class(self.solver_params.clone(), *dataset.records(), kernel, nu)
            }
            None => panic!("One class needs Nu value"),
        }
    }
}

/// Predict a probability with a feature vector
impl<'a, F: Float> Predict<Array1<F>, Pr> for Svm<F, Pr> {
    fn predict(&self, data: Array1<F>) -> Pr {
        let val = self.weighted_sum(&data) - self.rho;
        // this is safe because `F` is only implemented for `f32` and `f64`
        Pr(val.to_f32().unwrap())
    }
}

/// Predict a probability with a feature vector
impl<'a, F: Float> Predict<ArrayView1<'a, F>, Pr> for Svm<F, Pr> {
    fn predict(&self, data: ArrayView1<'a, F>) -> Pr {
        let val = self.weighted_sum(&data) - self.rho;
        // this is safe because `F` is only implemented for `f32` and `f64`
        Pr(val.to_f32().unwrap())
    }
}

/// Classify observations
///
/// This function takes a number of features and predicts target probabilities that they belong to
/// the positive class.
impl<'a, F: Float, D: Data<Elem = F>> Predict<ArrayBase<D, Ix2>, Array1<Pr>> for Svm<F, Pr> {
    fn predict(&self, data: ArrayBase<D, Ix2>) -> Array1<Pr> {
        data.outer_iter()
            .map(|data| {
                let val = self.weighted_sum(&data) - self.rho;
                // this is safe because `F` is only implemented for `f32` and `f64`
                Pr(val.to_f32().unwrap())
            })
            .collect()
    }
}

impl<'a, F: Float, D: Data<Elem = F>, T: Targets>
    Predict<DatasetBase<ArrayBase<D, Ix2>, T>, DatasetBase<ArrayBase<D, Ix2>, Array1<Pr>>>
    for Svm<F, Pr>
{
    fn predict(
        &self,
        data: DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> DatasetBase<ArrayBase<D, Ix2>, Array1<Pr>> {
        let DatasetBase { records, .. } = data;
        let predicted = self.predict(records.view());

        DatasetBase::new(records, predicted)
    }
}

impl<'a, F: Float, T: Targets, D: Data<Elem = F>>
    Predict<&'a DatasetBase<ArrayBase<D, Ix2>, T>, DatasetBase<ArrayView2<'a, F>, Array1<Pr>>>
    for Svm<F, Pr>
{
    fn predict(
        &self,
        data: &'a DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> DatasetBase<ArrayView2<'a, F>, Array1<Pr>> {
        let predicted = self.predict(data.records.view());

        DatasetBase::new(data.records.view(), predicted)
    }
}

#[cfg(test)]
mod tests {
    use super::Svm;
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
    fn test_linear_classification() {
        let entries: Array2<f64> = ndarray::stack(
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
        let model = Svm::params()
            .pos_neg_weights(1.0, 1.0)
            .linear_kernel()
            .fit(&dataset);

        let valid = model
            .predict(DatasetBase::from(entries))
            .map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert_eq!(cm.accuracy(), 1.0);

        // train model with Nu parameter
        let model = Svm::params().nu_weight(0.05).linear_kernel().fit(&dataset);

        let valid = model.predict(&valid).map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert_eq!(cm.accuracy(), 1.0);
    }

    #[test]
    fn test_polynomial_classification() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        // construct parabolica and classify middle area as positive and borders as negative
        let records = Array::random_using((40, 1), Uniform::new(-2f64, 2.), &mut rng);
        let targets = records.map_axis(Axis(1), |x| x[0] * x[0] < 0.5);
        let dataset = Dataset::new(records.clone(), targets);

        // train model with positive and negative weight
        let model = Svm::params()
            .pos_neg_weights(1.0, 1.0)
            .polynomial_kernel(0.0, 2.0)
            .fit(&dataset);

        //println!("{:?}", model.predict(DatasetBase::from(records.clone())).targets());

        let valid = model
            .predict(DatasetBase::from(records))
            .map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert!(cm.accuracy() > 0.9);
    }

    #[test]
    fn test_convoluted_rings_classification() {
        let records = generate_convoluted_rings(10);
        let targets = (0..20).map(|x| x < 10).collect::<Array1<_>>();
        let dataset = (records.view(), targets.view()).into();

        // train model with positive and negative weight
        let model = Svm::params()
            .pos_neg_weights(1.0, 1.0)
            .gaussian_kernel(50.0)
            .fit(&dataset);

        let valid = model
            .predict(DatasetBase::from(records.view()))
            .map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert!(cm.accuracy() > 0.9);

        // train model with Nu parameter
        let model = Svm::params()
            .nu_weight(0.01)
            .gaussian_kernel(50.0)
            .fit(&dataset);

        let valid = model.predict(&valid).map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert!(cm.accuracy() > 0.9);
    }

    #[test]
    fn test_reject_classification() {
        // generate two clusters with 100 samples each
        let entries = Array::random((100, 2), Uniform::new(-4., 4.));
        let dataset = DatasetBase::new(entries.view(), ());

        // train model with positive and negative weight
        let model = Svm::params()
            .nu_weight(1.0)
            .gaussian_kernel(100.0)
            .fit(&dataset);

        let valid = DatasetBase::from(Array::random((100, 2), Uniform::new(-10., 10f32)));
        let valid = model.predict(valid).map_targets(|x| **x > 0.0);

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
    }
}
