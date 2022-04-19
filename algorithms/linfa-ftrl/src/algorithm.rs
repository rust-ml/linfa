use crate::error::FtrlError;
use crate::hyperparams::{FtrlParams, FtrlValidParams};
use crate::FollowTheRegularizedLeader;
use linfa::dataset::{AsSingleTargets, Pr, Records};
use linfa::traits::{FitWith, PredictInplace};
use linfa::{DatasetBase, Float};
use ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix2, Zip};

/// Simplified `Result` using [`FTRLError`](crate::FTRLError) as error type
pub type Result<T> = std::result::Result<T, FtrlError>;

impl<'a, F, D, T> FitWith<'a, ArrayBase<D, Ix2>, T, FtrlError> for FtrlValidParams<F>
where
    F: Float,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = bool>,
{
    type ObjectIn = Option<FollowTheRegularizedLeader<F>>;
    type ObjectOut = FollowTheRegularizedLeader<F>;

    /// Fit a follow the regularized leader, proximal, model given a feature matrix `x` and a target
    /// variable `y`.
    ///
    /// The feature matrix `x` must have shape `(n_samples, n_features)`
    ///
    /// The target variable `y` must have shape `(n_samples)`
    ///
    /// Returns a fitted `FollowTheRegularizedLeader` object which contains the fitted
    /// parameters and can be used to `predict` values of the target variable
    /// for new feature values.
    fn fit_with(
        &self,
        model_in: Self::ObjectIn,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::ObjectOut> {
        let mut model_out = model_in.unwrap_or_else(|| {
            FollowTheRegularizedLeader::new(&FtrlParams(self.clone()), dataset.nfeatures())
        });
        let probabilities = model_out.predict_probabilities(dataset.records());
        let gradient = calculate_gradient(probabilities.view(), dataset);
        let sigma = model_out.calculate_sigma(gradient.view());
        model_out.update_params(gradient, sigma);
        Ok(model_out)
    }
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<Pr>>
    for FollowTheRegularizedLeader<F>
{
    /// Given an input matrix `X`, with shape `(n_samples, n_features)`,
    /// `predict` returns the target variable according to the parameters
    /// learned from the training data distribution.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<Pr>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        assert_eq!(
            x.ncols(),
            self.z.len(),
            "Number of data features must match the number of features the model was trained with."
        );

        let probabilities = self.predict_probabilities(x);
        Zip::from(&probabilities).and(y).for_each(|prob, out| {
            *out = Pr(F::to_f32(prob).unwrap_or_default());
        });
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<Pr> {
        Array1::zeros(x.nrows()).mapv(Pr)
    }
}

/// View the fitted parameters and make predictions with a fitted
/// follow the regularized leader -proximal, model
impl<F: Float> FollowTheRegularizedLeader<F> {
    /// Get Z values
    pub fn z(&self) -> &Array1<F> {
        &self.z
    }

    /// Get N values
    pub fn n(&self) -> &Array1<F> {
        &self.n
    }

    /// Get the hyperparameters
    pub fn get_params(&self) -> &FtrlParams<F> {
        &self.params
    }

    /// Calculate weights for model prediction
    pub fn get_weights(&self) -> Array1<F> {
        Zip::from(self.z.view())
            .and(self.n.view())
            .map_collect(|z_, n_| {
                calculate_weight(
                    *z_,
                    *n_,
                    self.params.0.alpha(),
                    self.params.0.beta(),
                    self.params.0.l1_ratio(),
                    self.params.0.l2_ratio(),
                )
            })
    }

    /// Update method of the model hyperparameters in case of async mode.
    /// To use this method, we must store probabilities and features for the row, and update once the result (bool) is observed.
    pub fn update<D: Data<Elem = F>, T: AsSingleTargets<Elem = bool>>(
        &mut self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>, probabilities: ArrayView1<Pr>) {
        let probabilities = probabilities.mapv(|prob| F::cast(prob.0));
        let gradient = calculate_gradient(probabilities.view(), dataset);
        let sigma = self.calculate_sigma(gradient.view());
        self.update_params(gradient, sigma);
    }

    fn predict_probabilities<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>) -> Array1<F> {
        let weights = self.get_weights();
        let mut probabilities = x.dot(&weights);
        probabilities.mapv_inplace(sigmoid);
        probabilities
    }

    fn calculate_sigma(&self, gradients: ArrayView1<F>) -> Array1<F> {
        Zip::from(&self.n)
            .and(gradients)
            .map_collect(|n_, grad| calculate_sigma(*n_, *grad, self.params.0.alpha))
    }

    fn update_params(&mut self, gradient: Array1<F>, sigma: Array1<F>) {
        self.z = &self.z + &gradient - sigma * self.get_weights();
        self.n = &self.n + gradient.mapv(|grad| grad.powf(F::cast(2.)));
    }
}

fn calculate_gradient<F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = bool>>(
    probabilities: ArrayView1<F>,
    dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
) -> Array1<F> {
    let targets = dataset.as_single_targets();
    let x = dataset.records();
    let diff = Zip::from(&probabilities)
        .and(targets)
        .map_collect(|prob, y| {
            let truth = if *y { F::one() } else { F::zero() };
            *prob - truth
        });
    diff.dot(x)
}

fn calculate_sigma<F: Float>(n: F, gradient: F, alpha: F) -> F {
    (F::sqrt(n + gradient.powf(F::cast(2.))) - F::sqrt(n)) / alpha
}

fn sigmoid<F: Float>(prediction: F) -> F {
    F::one() / (F::one() + F::exp(-F::max(F::min(prediction, F::cast(35.)), F::cast(-35.))))
}

fn calculate_weight<F: Float>(z: F, n: F, alpha: F, beta: F, l1_ratio: F, l2_ratio: F) -> F {
    let sign = if z < F::zero() { -F::one() } else { F::one() };
    if z * sign <= l1_ratio {
        F::zero()
    } else {
        (sign * l1_ratio - z) / ((n.sqrt() + beta) / alpha + l2_ratio)
    }
}

#[cfg(test)]
mod test {
    extern crate linfa;
    use super::*;
    use crate::algorithm::test::linfa::prelude::Predict;
    use approx::assert_abs_diff_eq;
    use linfa::Dataset;
    use ndarray::array;

    #[test]
    fn sigmoid_works() {
        let value = 100.;
        let result = sigmoid(value);
        assert!(result > 0.9)
    }

    #[test]
    fn calculate_weights_with_zero_outcome_works() {
        let z = 0.1;
        let n = 0.1;
        let alpha = 0.5;
        let beta = 0.5;
        let l1_ratio = 0.5;
        let l2_ratio = 0.5;
        let result = calculate_weight(z, n, alpha, beta, l1_ratio, l2_ratio);
        assert_abs_diff_eq!(result, 0.0)
    }

    #[test]
    fn calculate_sigma_works() {
        let gradient: f64 = 0.5;
        let n: f64 = 0.11;
        let alpha = 0.5;
        let expected_result = (((0.11 + 0.25) as f64).sqrt() - (0.11 as f64).sqrt()) / 0.5;
        let result = calculate_sigma(n, gradient, alpha);
        assert_abs_diff_eq!(result, expected_result)
    }

    #[test]
    fn calculate_weights_works() {
        let z = 0.5;
        let n: f64 = 0.16;
        let alpha = 0.5;
        let beta = 0.5;
        let l1_ratio = 0.1;
        let l2_ratio = 0.5;
        let expected_result = (0.1 - 0.5) / ((0.4 + 0.5) / 0.5 + 0.5);
        let result = calculate_weight(z, n, alpha, beta, l1_ratio, l2_ratio);
        assert_abs_diff_eq!(result, expected_result)
    }

    #[test]
    fn calculate_gradient_works() {
        let probabilities = array![0.1, 0.3, 0.8];
        let dataset = Dataset::new(
            array![[0.0, 1.0], [2.0, 3.0], [1.0, 5.0]],
            array![false, false, true],
        );
        let result = calculate_gradient(probabilities.view(), &dataset);
        assert_abs_diff_eq!(result, array![0.4, 0.0])
    }

    #[test]
    fn update_params_works() {
        let probabilities = array![0.1, 0.3, 0.8];
        let dataset = Dataset::new(
            array![[0.0, 1.0], [2.0, 3.0], [1.0, 5.0]],
            array![false, false, true],
        );
        let params = FtrlParams::default();
        let mut model = FollowTheRegularizedLeader::new(&params, dataset.nfeatures());
        let initial_z = model.z().clone();
        let initial_n = model.n().clone();
        let weights = model.get_weights();
        let gradient = calculate_gradient(probabilities.view(), &dataset);
        let sigma = model.calculate_sigma(gradient.view());
        model.update_params(gradient.clone(), sigma.clone());
        let expected_z = initial_z + &gradient - sigma * weights;
        let expected_n = initial_n + &gradient.mapv(|grad| (grad as f64).powf(2.));
        assert_abs_diff_eq!(model.z(), &expected_z);
        assert_abs_diff_eq!(model.n(), &expected_n)
    }

    #[test]
    fn predict_probabilities_works() {
        let dataset = Dataset::new(
            array![[0.0, 1.0], [2.0, 3.0], [1.0, 5.0]],
            array![false, false, true],
        );
        let params = FtrlParams::default();
        let model = FollowTheRegularizedLeader::new(&params, dataset.nfeatures());
        let probabilities = model.predict_probabilities(dataset.records());
        assert!(probabilities
            .iter()
            .all(|prob| prob >= &0.0 && prob <= &1.0));
    }

    #[test]
    fn update_works() {
        let probabilities = array![0.5, 0.3, 0.7].mapv(Pr);
        let dataset = Dataset::new(
            array![[0.0, 1.0], [2.0, 3.0], [1.0, 5.0]],
            array![false, false, true],
        );

        // Initialize model this way to control random z values
        let mut model = FollowTheRegularizedLeader {
            params: FtrlParams::default(),
            z: array![0.5, 0.7],
            n: array![0.0, 0.0],
        };
        model.update(&dataset, probabilities.view());
        assert_abs_diff_eq!(
            model.n(),
            &array![0.09, 0.01],
            epsilon = 1e-2
        );
        assert_abs_diff_eq!(
            model.z(),
            &array![0.8, 8.6],
            epsilon = 1e-2
        );
    }

    #[test]
    fn ftrl_toy_example_works() {
        let dataset = Dataset::new(
            array![[-1.0], [-2.0], [10.0], [9.0]],
            array![true, true, false, false],
        );
        let params = FollowTheRegularizedLeader::params()
            .l2_ratio(0.5)
            .l1_ratio(0.5)
            .alpha(0.1)
            .beta(0.0);

        // Initialize model this way to control random z values
        let model = FollowTheRegularizedLeader {
            params: params.clone(),
            z: array![0.5],
            n: array![0.],
        };
        let model = params.fit_with(Some(model), &dataset).unwrap();
        let test_x = array![[11.0]];
        assert_abs_diff_eq!(
            model.predict(&test_x).mapv(|v| v.0),
            array![0.25],
            epsilon = 1e-2
        );
    }

    #[test]
    fn ftrl_2d_toy_example_works() {
        let dataset = Dataset::new(array![[0.0, -5.0], [10.0, 20.0]], array![true, false]);
        let params = FollowTheRegularizedLeader::params()
            .l2_ratio(0.5)
            .l1_ratio(0.5)
            .alpha(0.01)
            .beta(0.0);

        // Initialize model this way to control random z values
        let model = FollowTheRegularizedLeader {
            params: params.clone(),
            z: array![0.5, 0.5],
            n: array![0.0, 0.0],
        };
        let model = params.fit_with(Some(model), &dataset).unwrap();
        let test_x = array![[-4.0, -10.0], [15.0, 25.0]];
        assert_abs_diff_eq!(
            model.predict(&test_x).mapv(|v| v.0),
            array![0.53, 0.401],
            epsilon = 1e-2
        );
    }
}
