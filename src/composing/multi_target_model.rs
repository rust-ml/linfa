//! Merge models with single target to multi-target models
//!
//! Many models assume that the target variables are uncorrelated and support therefore only a
//! single target variable. This wrapper allows the user to merge multiple models with only a
//! single-target variable into a multi-target model.
//!
//!
use crate::dataset::Records;
use crate::traits::PredictInplace;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use std::iter::FromIterator;

/// Merge models with single target to multi-target models
///
/// Many models assume that the target variables are uncorrelated and support therefore only a
/// single target variable. This wrapper allows the user to merge multiple models with only a
/// single-target variable into a multi-target model.
pub struct MultiTargetModel<R: Records, L> {
    models: Vec<Box<dyn PredictInplace<R, Array1<L>>>>,
}

impl<R: Records, L> MultiTargetModel<R, L> {
    /// Create a wrapper model from a list of single-target models
    ///
    /// The type parameter of the single-target models are only constraint to implement the
    /// prediction trait and can otherwise contain any object. This allows the mixture of different
    /// models into the same wrapper. If you want to use the same model for all predictions, just
    /// use the `FromIterator` implementation.
    pub fn new(models: Vec<Box<dyn PredictInplace<R, Array1<L>>>>) -> Self {
        MultiTargetModel { models }
    }
}

impl<L: Default, F, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array2<L>>
    for MultiTargetModel<ArrayBase<D, Ix2>, L>
{
    fn predict_inplace(&self, arr: &ArrayBase<D, Ix2>, targets: &mut Array2<L>) {
        assert_eq!(
            targets.shape(),
            &[arr.nrows(), self.models.len()],
            "The number of data points must match the number of output targets."
        );
        *targets = self
            .models
            .iter()
            .flat_map(|model| {
                let mut targets = Array1::default(arr.nrows());
                model.predict_inplace(arr, &mut targets);
                targets.into_raw_vec()
            })
            .collect::<Array1<L>>()
            .into_shape_with_order((self.models.len(), arr.len_of(Axis(0))))
            .unwrap()
            .reversed_axes();
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<L> {
        Array2::default((x.nrows(), self.models.len()))
    }
}

impl<F, D: Data<Elem = F>, L, P: PredictInplace<ArrayBase<D, Ix2>, Array1<L>> + 'static>
    FromIterator<P> for MultiTargetModel<ArrayBase<D, Ix2>, L>
{
    fn from_iter<I: IntoIterator<Item = P>>(iter: I) -> Self {
        let models = iter
            .into_iter()
            .map(|x| Box::new(x) as Box<dyn PredictInplace<ArrayBase<D, Ix2>, Array1<L>>>)
            .collect();

        MultiTargetModel { models }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        traits::{Predict, PredictInplace},
        MultiTargetModel,
    };
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2, Axis};

    /// First dummy model, returns a constant value
    struct DummyModel {
        val: f32,
    }

    impl PredictInplace<Array2<f32>, Array1<f32>> for DummyModel {
        fn predict_inplace(&self, arr: &Array2<f32>, targets: &mut Array1<f32>) {
            assert_eq!(
                arr.nrows(),
                targets.len(),
                "The number of data points must match the number of output targets."
            );
            targets.fill(self.val);
        }

        fn default_target(&self, x: &Array2<f32>) -> Array1<f32> {
            Array1::zeros(x.nrows())
        }
    }

    /// Second dummy model, counts up from a start value to the number of samples
    struct DummyModel2 {
        val: f32,
    }

    impl PredictInplace<Array2<f32>, Array1<f32>> for DummyModel2 {
        fn predict_inplace(&self, arr: &Array2<f32>, targets: &mut Array1<f32>) {
            assert_eq!(
                arr.nrows(),
                targets.len(),
                "The number of data points must match the number of output targets."
            );
            *targets = Array1::linspace(
                self.val,
                self.val + arr.len_of(Axis(0)) as f32 - 1.0,
                arr.len_of(Axis(0)),
            );
        }

        fn default_target(&self, x: &Array2<f32>) -> Array1<f32> {
            Array1::zeros(x.nrows())
        }
    }

    #[test]
    fn dummy_constant() {
        // construct models which predicts a constant all time
        // and merge them into a `MultiTargetModel`
        let model = (0..4)
            .map(|val| val as f32)
            .map(|val| DummyModel { val })
            .collect::<MultiTargetModel<_, _>>();

        // test capability to predict constants
        let targets = model.predict(&Array2::zeros((5, 2)));
        assert_abs_diff_eq!(
            targets,
            array![
                [0., 1., 2., 3.],
                [0., 1., 2., 3.],
                [0., 1., 2., 3.],
                [0., 1., 2., 3.],
                [0., 1., 2., 3.],
            ]
        );
    }

    #[test]
    fn different_dummys() {
        // create two different models, the first predicts a constant 42 and the second counts up
        // from 42 to the number of samples
        let model_a = DummyModel { val: 42.0 };
        let model_b = DummyModel2 { val: 42.0 };

        let model = MultiTargetModel::new(vec![Box::new(model_a), Box::new(model_b)]);

        let targets = model.predict(&Array2::zeros((5, 2)));
        assert_abs_diff_eq!(
            targets,
            array![[42., 42.], [42., 43.], [42., 44.], [42., 45.], [42., 46.]]
        );
    }
}
