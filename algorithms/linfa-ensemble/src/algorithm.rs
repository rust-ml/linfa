use crate::EnsembleLearnerValidParams;
use linfa::{
    dataset::{AsTargets, AsTargetsMut, FromTargetArrayOwned, Records},
    error::Error,
    traits::*,
    DatasetBase,
};
use ndarray::{Array2, Axis, Zip};
use rand::Rng;
use std::{cmp::Eq, collections::HashMap, hash::Hash};

pub struct EnsembleLearner<M> {
    pub models: Vec<M>,
    pub model_features: Vec<Vec<usize>>,
}

impl<M> EnsembleLearner<M> {
    // Generates prediction iterator returning predictions from each model
    pub fn generate_predictions<'b, R: Records, T>(
        &'b self,
        x: &'b Vec<R>,
    ) -> impl Iterator<Item = T> + 'b
    where
        M: Predict<&'b R, T>,
    {
        self.models
            .iter()
            .zip(x.into_iter())
            .map(move |(m, sub_data)| m.predict(sub_data))
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

        let sub_datas = self
            .model_features
            .iter()
            .map(|feat| x.select(Axis(1), feat))
            .collect::<Vec<_>>();
        let predictions = self.generate_predictions(&sub_datas);

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
        let mut models = Vec::with_capacity(self.ensemble_size);
        let mut model_features = Vec::with_capacity(self.ensemble_size);
        let mut rng = self.rng.clone();

        // Compute dataset and the subset of features ratio to be selected
        let dataset_size =
            ((dataset.records.nrows() as f64) * self.bootstrap_proportion).ceil() as usize;
        let n_feat = dataset.records.ncols();
        let n_sub = ((n_feat as f64) * self.feature_proportion).ceil() as usize;

        let iter = dataset.bootstrap_with_indices((dataset_size, n_sub), &mut rng);
        for (train, _, feature_selected) in iter {
            let model = self.model_params.fit(&train).unwrap();
            models.push(model);
            model_features.push(feature_selected);

            if models.len() == self.ensemble_size {
                break;
            }
        }

        Ok(EnsembleLearner {
            models: models,
            model_features: model_features,
        })
    }
}
