//! Merge models with binary to multi-class classification
//!
use crate::dataset::{Records, Pr};
use crate::traits::PredictRef;
use crate::Float;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use std::iter::FromIterator;

pub struct MultiClassModel<R: Records, L> {
    models: Vec<(L, Box<dyn PredictRef<R, Array1<Pr>>>)>,
}

impl<R: Records, L> MultiClassModel<R, L> {
    pub fn new(models: Vec<(L, Box<dyn PredictRef<R, Array1<Pr>>>)>) -> Self {
        MultiClassModel { models }
    }
}

impl<L: Clone + Default, F: Float, D: Data<Elem = F>> PredictRef<ArrayBase<D, Ix2>, Array1<L>>
    for MultiClassModel<ArrayBase<D, Ix2>, L>
{
    fn predict_ref(&self, arr: &ArrayBase<D, Ix2>) -> Array1<L> {
        let mut res = Array1::from_elem(arr.nrows(), L::default());

        /*self.models
            .iter()
            .flat_map(|model| model.predict_ref(arr).into_raw_vec())
            .collect::<Array1<L>>()
            .into_shape((self.models.len(), arr.len_of(Axis(0))))
            .unwrap()
            .reversed_axes()*/
        panic!("");
    }
}

/*
impl<F: Float, D: Data<Elem = F>, L, P: PredictRef<ArrayBase<D, Ix2>, Array1<L>> + 'static>
    FromIterator<P> for MultiTargetModel<ArrayBase<D, Ix2>, L>
{
    fn from_iter<I: IntoIterator<Item = P>>(iter: I) -> Self {
        let models = iter
            .into_iter()
            .map(|x| Box::new(x) as Box<dyn PredictRef<ArrayBase<D, Ix2>, Array1<L>>>)
            .collect();

        MultiTargetModel { models }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        traits::{Predict, PredictRef},
        MultiTargetModel,
    };
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2, Axis};

    /// First dummy model, returns a constant value
    struct DummyModel {
        val: f32,
    }

    impl PredictRef<Array2<f32>, Array1<f32>> for DummyModel {
        fn predict_ref(&self, arr: &Array2<f32>) -> Array1<f32> {
            Array1::from_elem(arr.len_of(Axis(0)), self.val)
        }
    }

    /// Second dummy model, counts up from a start value to the number of samples
    struct DummyModel2 {
        val: f32,
    }

    impl PredictRef<Array2<f32>, Array1<f32>> for DummyModel2 {
        fn predict_ref(&self, arr: &Array2<f32>) -> Array1<f32> {
            Array1::linspace(
                self.val,
                self.val + arr.len_of(Axis(0)) as f32 - 1.0,
                arr.len_of(Axis(0)),
            )
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
*/
