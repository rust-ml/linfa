use linfa::{
    dataset::{AsTargets, AsTargetsMut, FromTargetArrayOwned, Records},
    error::{Error},
    traits::*,
    DatasetBase,
};
use ndarray::{
    Array2, Axis, Array, Dimension
};
use std::{
    cmp::Eq,
    collections::HashMap,
    hash::Hash,
};
use rand::Rng;
use rand::rngs::ThreadRng;

pub struct EnsembleLearner<M> {
    pub models: Vec<M>,
}

impl<M> EnsembleLearner<M> {

    // Generates prediction iterator returning predictions from each model
    pub fn generate_predictions<'b, R: Records, T>(&'b self, x: &'b R) -> impl Iterator<Item = T> + 'b
    where M: Predict<&'b R, T> {
        self.models.iter().map(move |m| m.predict(x))
    }

    // Consumes prediction iterator to return all predictions
    pub fn aggregate_predictions<Ys: Iterator>(&self, ys: Ys)
    -> impl Iterator<Item = Vec<(Array<<Ys::Item as AsTargets>::Elem, <<Ys::Item as AsTargets>::Ix as Dimension>::Smaller >, usize)>>
    where
        Ys::Item: AsTargets,
        <Ys::Item as AsTargets>::Elem: Copy + Eq + Hash,
    {
        let mut prediction_maps = Vec::new();

        for y in ys {
            let targets = y.as_targets();
            let no_targets = targets.shape()[0];

            for i in 0..no_targets {
                if prediction_maps.len() == i {
                    prediction_maps.push(HashMap::new());
                }
                *prediction_maps[i].entry(y.as_targets().index_axis(Axis(0), i).to_owned()).or_insert(0) += 1;
            }
        }

        prediction_maps.into_iter().map(|xs| {
            let mut xs: Vec<_> = xs.into_iter().collect();
            xs.sort_by(|(_, x), (_, y)| y.cmp(x));
            xs
        })
    }

}

impl<F: Clone, T, M>
PredictInplace<Array2<F>, T> for EnsembleLearner<M>
where
    M: PredictInplace<Array2<F>, T>,
    <T as AsTargets>::Elem: Copy + Eq + Hash,
    T: AsTargets + AsTargetsMut<Elem = <T as AsTargets>::Elem>,
{
    fn predict_inplace(&self, x: &Array2<F>, y: &mut T) {
        let mut y_array = y.as_targets_mut();
        assert_eq!(
            x.nrows(),
            y_array.len_of(Axis(0)),
            "The number of data points must match the number of outputs."
        );

        let mut predictions = self.generate_predictions(x);
        let aggregated_predictions = self.aggregate_predictions(&mut predictions);

        for (target, output) in y_array.axis_iter_mut(Axis(0)).zip(aggregated_predictions.into_iter()) {
            for (t, o) in target.into_iter().zip(output[0].0.iter()) {
                *t = *o;
            }
        }
    }

    fn default_target(&self, x: &Array2<F>) -> T {
        self.models[0].default_target(x)
    }
}

pub struct EnsembleLearnerParams<P, R: Rng + Clone> {
    pub ensemble_size: usize,
    pub bootstrap_proportion: f64,
    pub model_params: P,
    pub rng: R
}

impl<P> EnsembleLearnerParams<P, ThreadRng> {
    pub fn new(model_params: P) -> EnsembleLearnerParams<P, ThreadRng> {
        return Self::new_fixed_rng(model_params, rand::thread_rng())
    }
}

impl<P, R: Rng + Clone> EnsembleLearnerParams<P, R> {
    pub fn new_fixed_rng(model_params: P, rng: R) -> EnsembleLearnerParams<P, R> {
        EnsembleLearnerParams {
            ensemble_size: 1,
            bootstrap_proportion: 1.0,
            model_params: model_params,
            rng: rng
        }
    }

    pub fn ensemble_size(&mut self, size: usize) -> &mut EnsembleLearnerParams<P, R> {
        assert!(size > 0, "ensemble_size cannot be less than 1. Ensembles must consist of at least one model.");
        self.ensemble_size = size;
        self
    }

    pub fn bootstrap_proportion(&mut self, proportion: f64) -> &mut EnsembleLearnerParams<P, R> {
        assert!(proportion > 0.0, "bootstrap_proportion must be greater than 0. Must provide some data to each model.");
        self.bootstrap_proportion = proportion;
        self
    }

}

impl<D, T, P: Fit<Array2<D>, T::Owned, Error>, R: Rng + Clone>
     Fit<Array2<D>, T, Error> for EnsembleLearnerParams<P, R>
where
    D: Clone,
    T: FromTargetArrayOwned,
    T::Elem: Copy + Eq + Hash,
    T::Owned: AsTargets,
{
    type Object = EnsembleLearner<P::Object>;

    fn fit(&self, dataset: &DatasetBase<Array2<D>, T>) -> Result<Self::Object, Error> {

        let mut models = Vec::new();
        let mut rng = self.rng.clone();

        let dataset_size = ((dataset.records.shape()[0] as f64) * self.bootstrap_proportion).ceil() as usize;

        let iter = dataset.bootstrap_samples(dataset_size, &mut rng);

        for train in iter {
            let model = self.model_params.fit(&train).unwrap();
            models.push(model);

            if models.len() == self.ensemble_size {
                break
            }
        }

        Ok(EnsembleLearner { models })
    }
}
