use linfa::{
    dataset::{AsTargets, AsTargetsMut, FromTargetArrayOwned, Records},
    error::{Error, Result},
    traits::*,
    DatasetBase, ParamGuard,
};
use ndarray::{Array, Array2, Axis, Dimension};
use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
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

    // Consumes prediction iterator to return all predictions
    pub fn aggregate_predictions<Ys: Iterator>(
        &self,
        ys: Ys,
    ) -> impl Iterator<
        Item = Vec<(
            Array<
                <Ys::Item as AsTargets>::Elem,
                <<Ys::Item as AsTargets>::Ix as Dimension>::Smaller,
            >,
            usize,
        )>,
    >
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
                *prediction_maps[i]
                    .entry(y.as_targets().index_axis(Axis(0), i).to_owned())
                    .or_insert(0) += 1;
            }
        }

        prediction_maps.into_iter().map(|xs| {
            let mut xs: Vec<_> = xs.into_iter().collect();
            xs.sort_by(|(_, x), (_, y)| y.cmp(x));
            xs
        })
    }
}

impl<F: Clone, T, M> PredictInplace<Array2<F>, T> for EnsembleLearner<M>
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

        for (target, output) in y_array
            .axis_iter_mut(Axis(0))
            .zip(aggregated_predictions.into_iter())
        {
            for (t, o) in target.into_iter().zip(output[0].0.iter()) {
                *t = *o;
            }
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
pub struct EnsembleLearnerParams<P, R>(pub EnsembleLearnerValidParams<P, R>);

impl<P> EnsembleLearnerParams<P, StdRng>
where
    StdRng: Send + Sync,
{
    pub fn new(model_params: P) -> EnsembleLearnerParams<P, StdRng> {
        return Self::new_fixed_rng(model_params, <StdRng as rand::SeedableRng>::from_entropy());
    }
}

impl<P, R: Rng + Clone + Send + Sync> EnsembleLearnerParams<P, R> {
    pub fn new_fixed_rng(model_params: P, rng: R) -> EnsembleLearnerParams<P, R> {
        Self(EnsembleLearnerValidParams {
            ensemble_size: 1,
            bootstrap_proportion: 1.0,
            model_params: model_params,
            rng: rng,
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

impl<D, T, P, R: Rng + Clone + Send + Sync> Fit<Array2<D>, T, Error>
    for EnsembleLearnerValidParams<P, R>
where
    D: Clone + Sync + Send,
    T: FromTargetArrayOwned + Sync,
    T::Elem: Copy + Eq + Hash,
    T::Owned: AsTargets + Send,
    P: Fit<Array2<D>, T::Owned, Error> + Sync + Send, // Ensure P is Send
    P::Object: Send,                                  // Ensure P::Object is Send
{
    type Object = EnsembleLearner<P::Object>;

    fn fit(
        &self,
        dataset: &DatasetBase<Array2<D>, T>,
    ) -> core::result::Result<Self::Object, Error> {
        let mut rng = self.rng.clone();

        let dataset_size =
            ((dataset.records.nrows() as f64) * self.bootstrap_proportion).ceil() as usize;

        let iter = dataset.parallel_bootstrap_samples(dataset_size, &mut rng);

        let count = AtomicUsize::new(0);

        // might potentially slow down the code
        // TODO: instead we can spawn threads that compute n_estimators/threads
        // and then join all the vectors
        let models: Vec<_> = iter
            .take_any_while(|_| count.load(Ordering::Relaxed) < self.ensemble_size)
            .map(|train| {
                let _c = count.fetch_add(1, Ordering::SeqCst);
                self.model_params.fit(&train).unwrap()
            })
            .collect();

        // println!("Done with fit: {}", models.len());

        Ok(EnsembleLearner { models })
    }
}

#[cfg(test)]
mod tests {
    use linfa_trees::DecisionTree;
    use ndarray_rand::rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    #[test]
    fn iris_test() {
        //Number of models in the ensemble
        let ensemble_size = 100;
        //Proportion of training data given to each model
        let bootstrap_proportion = 0.7;

        //Load dataset
        let mut rng = SmallRng::seed_from_u64(42);
        let (train, test) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.7);

        //Train ensemble learner model
        let model = EnsembleLearnerParams::new(DecisionTree::<f64, usize>::params())
            .ensemble_size(ensemble_size)
            .bootstrap_proportion(bootstrap_proportion)
            .fit(&train)
            .unwrap();
        // println!("Done with Fit");
        //   //Return highest ranking predictions
        let final_predictions_ensemble = model.predict(&test);
        println!("Final Predictions: \n{:?}", final_predictions_ensemble);

        //     let cm = final_predictions_ensemble.confusion_matrix(&test).unwrap();

        //     println!("{:?}", cm);
        //     println!("Test accuracy: {} \n with default Decision Tree params, \n Ensemble Size: {},\n Bootstrap Proportion: {}",
        //   100.0 * cm.accuracy(), ensemble_size, bootstrap_proportion);
    }
}
