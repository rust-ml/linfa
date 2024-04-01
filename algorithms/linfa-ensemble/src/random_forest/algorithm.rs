// use std::collections::HashMap;

// use linfa_trees::*;
// use linfa_datasets;
// use linfa::{prelude::*, Label, error::Error, error::Result};
// use linfa::{self, traits::Fit};
// // use linfa::dataset;
// // use super::super::linfa::dataset;
// use ndarray_rand::rand::SeedableRng;
// use rand::rngs::SmallRng;
// use rand::{seq::SliceRandom, Rng};

// use linfa::dataset::{AsSingleTargets, FromTargetArray, Labels, Records};
// use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};

// use super::RandomForestParams;
// // use crate::RandomForestParams;

// #[derive(Debug, Clone, PartialEq)]
// pub struct RandomForest<F: Float, L: Label> {
//   trees: Vec<DecisionTree<F, L>>,

// }

// impl<F: Float, L: Label + Default, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
//     for RandomForest<F, L>
// {
//     /// Make predictions for each row of a matrix of features `x`.
//     fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
//         assert_eq!(
//             x.nrows(),
//             y.len(),
//             "The number of data points must match the number of output targets."
//         );
        
//         let mut map: HashMap<usize, HashMap<L, usize>> = HashMap::new();
//         for tree in self.trees.iter() {
//             // go over each data row and aggregate the weights of the stump in a hashmap
//             for (idx, pred) in tree.predict(x).iter().enumerate() {
//                 let map_labels = map.entry(idx).or_insert(HashMap::new());
//                 *map_labels.entry(pred.clone()).or_default() += 1;
//             }
//         }

//         // set the label with maximum weight in the target "y"
//         for (idx, target) in y.iter_mut().enumerate() {
//             let mut max_entry = None;
//             let map_labels = map.get(&idx).unwrap();
//             // find the max value in map_labels
//             for (key, value) in map_labels.iter() {
//                 // Check if this entry has a greater value than the current maximum
//                 if max_entry.map_or(true, |(_, max_value)| value > max_value) {
//                     // Update the maximum entry
//                     max_entry = Some((key, value));
//                 }
//             }
//             *target = max_entry.unwrap().0.clone();
//         }
//     }

//     fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
//         Array1::default(x.nrows())
//     }
// }


// impl<F: Float, L, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
//     for RandomForestParams<F, L>
// where
//     L: Label + std::fmt::Debug + Copy,
//     D: Data<Elem = F>,
//     T: AsSingleTargets<Elem = L> + Labels<Elem = L>,// + FromTargetArray<'a, Elem = L>,
//     // T::Owned: AsTargets,
//     // <<T as FromTargetArray<'a>>::Owned as AsTargets>::Elem: Label,
// {
//     type Object = RandomForest<F, L>;

    
//     fn fit<'b>(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {

//       let mut rng = SmallRng::seed_from_u64(42);
//       //   dataset.bootstrap((10, 10), &mut rng); // doesn't work due to life time constraints
//         let mut trees = Vec::new();

//         for _i in 0..self.n_estimators() {
          
//           // let sample_feature_size = (150, 2);
//           // // sample with replacement
//           // let indices = (0..sample_feature_size.0)
//           // .map(|_| rng.gen_range(0..dataset.nsamples()))
//           // .collect::<Vec<_>>();
  
//           // let records = dataset.records().select(Axis(0), &indices);
//           // let targets = T::new_targets(dataset.as_targets().select(Axis(0), &indices));
  
//           // let indices = (0..sample_feature_size.1)
//           //     .map(|_| rng.gen_range(0..dataset.nfeatures()))
//           //     .collect::<Vec<_>>();
  
//           // let records = dataset.records().select(Axis(0), &indices);
  
//           let temp_data = bootstrap_fn(dataset);

//           let gini_model = DecisionTree::params()
//             .split_quality(SplitQuality::Gini)
//             .max_depth(Some(100))
//             .min_weight_split(1.0)
//             .min_weight_leaf(1.0)
//             .fit(&temp_data)?;

//           trees.push(gini_model);
          
//           // Determine the number of bootstrap samples based on your parameters.
//           // For illustration, let's use a simple fixed size.
//           // let num_bootstrap_samples = self.n_estimators();
  
//           // // let mut forest = Vec::with_capacity(num_bootstrap_samples);
  
//           // // Use the bootstrap iterator within this scope.
//           // for _ in 1..10 {
  
//           //   dataset.clone().bootstrap((10, 10), &mut rng).next().unwrap();
//           // }
//         }

//         Ok(RandomForest{trees})

//         // todo!() // return
//     }
// }

// // impl<F: Float, L: Label> RandomForest<F, L> 
// // {
//     // fn bootstrap_fn<'a, F, L, D, T>(dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> DatasetBase<Array2<F>, <T as FromTargetArray<'a>>::Owned>//DatasetBase<ArrayBase<D, Ix2>, T> 
//     // where
//     //   F: Float,
//     //   L: Label + std::fmt::Debug + Copy,
//     //   D: Data<Elem = F>,
//     //   T: AsSingleTargets<Elem = L> + Labels<Elem = L> + FromTargetArray<'a, Elem = L>,
//     //   T::Owned: AsTargets, 
//     // {
//     //   let mut rng = SmallRng::seed_from_u64(42);
//     //   let sample_feature_size = (150, 2);
//     //   // sample with replacement
//     //   let indices = (0..sample_feature_size.0)
//     //   .map(|_| rng.gen_range(0..dataset.nsamples()))
//     //   .collect::<Vec<_>>();

//     //   let records = dataset.records().select(Axis(0), &indices);
//     //   let targets = T::new_targets(dataset.as_targets().select(Axis(0), &indices));

//     //   let indices = (0..sample_feature_size.1)
//     //       .map(|_| rng.gen_range(0..dataset.nfeatures()))
//     //       .collect::<Vec<_>>();

//     //   let records = records.select(Axis(1), &indices);
//     //   DatasetBase::new(records, targets)
//     //   // let temp_data = 

//     //   // return temp_data;
//     // }
// // }

// // fn max_voted<F: Float, L: Label> (
// //   row: &ArrayBase<impl Data<Elem = F>, Ix1>,
// //   trees: &Vec<DecisionTree<F, L>>
// // ) -> L {
// //   let mut classes: HashMap<L, usize> = HashMap::new();

// //   for a_tree in trees.iter() {
    
// //   }
// // }

// pub fn testing() {
//   // Load the dataset
//   let dataset = linfa_datasets::iris();
//   // load Iris dataset
//   let mut rng = SmallRng::seed_from_u64(42);
//   let (train, test) = linfa_datasets::iris()
//   .shuffle(&mut rng)
//   .split_with_ratio(0.8);
//   // Fit the tree
//   // println!("Training model with Gini criterion ...");
//   //   let gini_model = DecisionTree::params()
//   //       .split_quality(SplitQuality::Gini)
//   //       .max_depth(Some(100))
//   //       .min_weight_split(1.0)
//   //       .min_weight_leaf(1.0)
//   //       .fit(&train)?;

//   //   let gini_pred_y = gini_model.predict(&test);
//   //   let cm = gini_pred_y.confusion_matrix(&test)?;

//   //   println!("{:?}", cm);

//   //   println!(
//   //       "Test accuracy with Gini criterion: {:.2}%",
//   //       100.0 * cm.accuracy()
//   //   );
//   // let tree = DecisionTree::params().fit(&dataset).unwrap();
//   // // Get accuracy on training set
//   // let accuracy = tree.predict(&dataset).confusion_matrix(&dataset).unwrap().accuracy();

//   // assert!(accuracy > 0.9);
// }

// fn main() {
//   testing();
// }

// #[cfg(test)]
// mod test_forest {
//   use super::*;

//     #[test]
//     fn it_works() {
//         testing();
//     }
// }


use linfa::{
  dataset::{AsTargets, AsTargetsMut, FromTargetArrayOwned, Records},
  error::{Error, Result},
  traits::*,
  DatasetBase, ParamGuard,
};
use ndarray::{Array, Array2, Axis, Dimension};
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
pub struct EnsembleLearnerParams<P, R>(EnsembleLearnerValidParams<P, R>);

impl<P> EnsembleLearnerParams<P, ThreadRng> {
  pub fn new(model_params: P) -> EnsembleLearnerParams<P, ThreadRng> {
      return Self::new_fixed_rng(model_params, rand::thread_rng());
  }
}

impl<P, R: Rng + Clone> EnsembleLearnerParams<P, R> {
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

      let iter = dataset.bootstrap_samples((dataset_size), &mut rng);
      let mut count = 0;
      for train in iter {
          count += 1;
          let model = self.model_params.fit(&train).unwrap();
          models.push(model);

          if models.len() == self.ensemble_size {
              break;
          }
      }
      println!("Total issss: {}", count);
      Ok(EnsembleLearner { models })
  }
}

#[cfg(test)]
mod test {
    use linfa::traits::Fit;
    use linfa_trees::DecisionTree;
    use rand::{rngs::SmallRng, SeedableRng};
    use linfa::prelude::{Predict, ToConfusionMatrix};
    use crate::EnsembleLearnerParams;


  #[test]
  fn run_test () {
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
    let model = EnsembleLearnerParams::new(DecisionTree::params())
        .ensemble_size(ensemble_size)
        .bootstrap_proportion(bootstrap_proportion)
        .fit(&train)
        .unwrap();

    //Return highest ranking predictions
    let final_predictions_ensemble = model.predict(&test);
    println!("Final Predictions: \n{:?}", final_predictions_ensemble);

    let cm = final_predictions_ensemble.confusion_matrix(&test).unwrap();

    println!("{:?}", cm);
    println!("Test accuracy: {} \n with default Decision Tree params, \n Ensemble Size: {},\n Bootstrap Proportion: {}",
    100.0 * cm.accuracy(), ensemble_size, bootstrap_proportion);
  }
}