//! Merge models with binary to multi-class classification
//!
use crate::dataset::{Pr, Records};
use crate::traits::PredictInplace;
use crate::Float;
use ndarray::{Array1, ArrayBase, Data, Ix2};
use std::iter::FromIterator;

type MultiClassVec<R, L> = Vec<(L, Box<dyn PredictInplace<R, Array1<Pr>>>)>;

/// Merge models with binary to multi-class classification
pub struct MultiClassModel<R: Records, L> {
    models: MultiClassVec<R, L>,
}

impl<R: Records, L> MultiClassModel<R, L> {
    pub fn new(models: MultiClassVec<R, L>) -> Self {
        MultiClassModel { models }
    }
}

impl<L: Clone + Default, F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
    for MultiClassModel<ArrayBase<D, Ix2>, L>
{
    fn predict_inplace(&self, arr: &ArrayBase<D, Ix2>, targets: &mut Array1<L>) {
        assert_eq!(
            arr.nrows(),
            targets.len(),
            "The number of data points must match the number of output targets."
        );

        let mut res = Vec::new();

        for pairs in self.models.iter().map(|(elm, model)| {
            let mut targets = Array1::default(arr.nrows());
            model.predict_inplace(arr, &mut targets);

            targets.into_iter().map(|x| (elm.clone(), *x)).collect()
        }) {
            // initialize result with guess of first model
            if res.is_empty() {
                res = pairs;
                continue;
            }

            // compare probability to each subsequent model and replace label
            // if probability is higher
            res = res
                .into_iter()
                .zip(pairs.into_iter())
                .map(|(c, d)| if d.1 > c.1 { d } else { c })
                .collect();
        }

        // remove probabilities from array and convert to `Array1`
        *targets = res.into_iter().map(|x| x.0).collect();
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

impl<
        F: Float,
        D: Data<Elem = F>,
        L,
        P: PredictInplace<ArrayBase<D, Ix2>, Array1<Pr>> + 'static,
    > FromIterator<(L, P)> for MultiClassModel<ArrayBase<D, Ix2>, L>
{
    fn from_iter<I: IntoIterator<Item = (L, P)>>(iter: I) -> Self {
        let models = iter
            .into_iter()
            .map(|(l, x)| {
                (
                    l,
                    Box::new(x) as Box<dyn PredictInplace<ArrayBase<D, Ix2>, Array1<Pr>>>,
                )
            })
            .collect();

        MultiClassModel { models }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dataset::Pr,
        traits::{Predict, PredictInplace},
        MultiClassModel,
    };
    use ndarray::{array, Array1, Array2};

    /// First dummy model, returns probability 1 for odd items
    struct DummyModel {
        on_even: bool,
    }

    impl PredictInplace<Array2<f32>, Array1<Pr>> for DummyModel {
        fn predict_inplace(&self, arr: &Array2<f32>, targets: &mut Array1<Pr>) {
            assert_eq!(
                arr.nrows(),
                targets.len(),
                "The number of data points must match the number of output targets."
            );

            if !self.on_even {
                *targets =
                    Array1::from_shape_fn(
                        arr.nrows(),
                        |x| if x % 2 == 1 { Pr(1.0) } else { Pr(0.0) },
                    );
            } else {
                *targets =
                    Array1::from_shape_fn(
                        arr.nrows(),
                        |x| if x % 2 == 1 { Pr(0.0) } else { Pr(1.0) },
                    );
            }
        }

        fn default_target(&self, x: &Array2<f32>) -> Array1<Pr> {
            Array1::default(x.nrows())
        }
    }

    #[test]
    fn correct_dummies() {
        let model1 = DummyModel { on_even: false };
        let model2 = DummyModel { on_even: true };

        let data = Array2::zeros((4, 2));
        assert_eq!(
            model1.predict(&data),
            array![Pr(0.0), Pr(1.0), Pr(0.0), Pr(1.0)]
        );
        assert_eq!(
            model2.predict(&data),
            array![Pr(1.0), Pr(0.0), Pr(1.0), Pr(0.0)]
        );
    }

    #[test]
    fn choose_correct() {
        let model = vec![
            (0, DummyModel { on_even: false }),
            (1, DummyModel { on_even: true }),
        ]
        .into_iter()
        .collect::<MultiClassModel<_, usize>>();

        let data = Array2::zeros((4, 2));
        assert_eq!(model.predict(&data), array![1, 0, 1, 0]);
    }
}
