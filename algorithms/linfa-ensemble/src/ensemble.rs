use linfa::{
    dataset::{AsTargets, AsTargetsMut, FromTargetArrayOwned, Records},
    error::{Error, Result},
    traits::*,
    DatasetBase, ParamGuard,
};
use ndarray::{Array2, Axis, Zip};
use rand::rngs::ThreadRng;
use rand::Rng;
use std::{cmp::Eq, collections::HashMap, hash::Hash};

pub struct EnsembleLearner<M> {
    pub models: Vec<M>,
}

impl<M> EnsembleLearner<M> {
    // Generates prediction iterator returning predictions from each model
    pub fn generate_predictions<'b, R: Records, T>(
        &'b self,
        x: &'b R,
    ) -> impl Iterator<Item = T> + 'b
    where
        M: Predict<&'b R, T>,
    {
        self.models.iter().map(move |m| m.predict(x))
    }
}

impl<F: Clone, T, M> PredictInplace<Array2<F>, T> for EnsembleLearner<M>
where
    M: PredictInplace<Array2<F>, T>,
    <T as AsTargets>::Elem: Copy + Eq + Hash + std::fmt::Debug,
    T: AsTargets + AsTargetsMut<Elem = <T as AsTargets>::Elem>,
{
    fn predict_inplace(&self, x: &Array2<F>, y: &mut T) {
        let y_array = y.as_targets();
        assert_eq!(
            x.nrows(),
            y_array.len_of(Axis(0)),
            "The number of data points must match the number of outputs."
        );

        let predictions = self.generate_predictions(x);

        // prediction map has same shape as y_array, but the elements are maps
        let mut prediction_maps = y_array.map(|_| HashMap::new());

        for prediction in predictions {
            let p_arr = prediction.as_targets();
            assert_eq!(p_arr.shape(), y_array.shape());
            // Insert each prediction value into the corresponding map
            Zip::from(&mut prediction_maps)
                .and(&p_arr)
                .for_each(|map, val| *map.entry(*val).or_insert(0) += 1);
        }

        // For each prediction, pick the result with the highest number of votes
        let agg_preds = prediction_maps.map(|map| map.iter().max_by_key(|(_, v)| **v).unwrap().0);
        let mut y_array = y.as_targets_mut();
        for (y, pred) in y_array.iter_mut().zip(agg_preds.iter()) {
            *y = **pred
        }
    }

    fn default_target(&self, x: &Array2<F>) -> T {
        self.models[0].default_target(x)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnsembleLearnerValidParams<P, R> {
    pub ensemble_size: usize,
    pub bootstrap_proportion: f64,
    pub model_params: P,
    pub rng: R,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnsembleLearnerParams<P, R>(EnsembleLearnerValidParams<P, R>);

impl<P> EnsembleLearnerParams<P, ThreadRng> {
    pub fn new(model_params: P) -> EnsembleLearnerParams<P, ThreadRng> {
        Self::new_fixed_rng(model_params, rand::thread_rng())
    }
}

impl<P, R: Rng + Clone> EnsembleLearnerParams<P, R> {
    pub fn new_fixed_rng(model_params: P, rng: R) -> EnsembleLearnerParams<P, R> {
        Self(EnsembleLearnerValidParams {
            ensemble_size: 1,
            bootstrap_proportion: 1.0,
            model_params,
            rng,
        })
    }

    pub fn ensemble_size(mut self, size: usize) -> Self {
        self.0.ensemble_size = size;
        self
    }

    pub fn bootstrap_proportion(mut self, proportion: f64) -> Self {
        self.0.bootstrap_proportion = proportion;
        self
    }
}

impl<P, R> ParamGuard for EnsembleLearnerParams<P, R> {
    type Checked = EnsembleLearnerValidParams<P, R>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.bootstrap_proportion > 1.0 || self.0.bootstrap_proportion <= 0.0 {
            Err(Error::Parameters(format!(
                "Bootstrap proportion should be greater than zero and less than or equal to one, but was {}",
                self.0.bootstrap_proportion
            )))
        } else if self.0.ensemble_size < 1 {
            Err(Error::Parameters(format!(
                "Ensemble size should be less than one, but was {}",
                self.0.ensemble_size
            )))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

impl<D, T, P: Fit<Array2<D>, T::Owned, Error>, R: Rng + Clone> Fit<Array2<D>, T, Error>
    for EnsembleLearnerValidParams<P, R>
where
    D: Clone,
    T: FromTargetArrayOwned,
    T::Elem: Copy + Eq + Hash,
    T::Owned: AsTargets,
{
    type Object = EnsembleLearner<P::Object>;

    fn fit(
        &self,
        dataset: &DatasetBase<Array2<D>, T>,
    ) -> core::result::Result<Self::Object, Error> {
        let mut models = Vec::new();
        let mut rng = self.rng.clone();

        let dataset_size =
            ((dataset.records.nrows() as f64) * self.bootstrap_proportion).ceil() as usize;

        let iter = dataset.bootstrap_samples(dataset_size, &mut rng);

        for train in iter {
            let model = self.model_params.fit(&train).unwrap();
            models.push(model);

            if models.len() == self.ensemble_size {
                break;
            }
        }

        Ok(EnsembleLearner { models })
    }
}
