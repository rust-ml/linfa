use linfa_trees::*;
use linfa_datasets;
// use linfa::prelude::*;
use linfa::{self, traits::Fit};
// use linfa::dataset;
// use super::super::linfa::dataset;
use ndarray_rand::rand::SeedableRng;
// use rand::rngs::SmallRng;
pub fn testing() {
  // Load the dataset
  // let dataset = linfa_datasets::iris();
  // // load Iris dataset
  // let mut rng = SmallRng::seed_from_u64(42);
  // let (train, test) = linfa_datasets::iris()
  // .shuffle(&mut rng)
  // .split_with_ratio(0.8);
  // // Fit the tree
  // println!("Training model with Gini criterion ...");
  //   let gini_model = DecisionTree::params()
  //       .split_quality(SplitQuality::Gini)
  //       .max_depth(Some(100))
  //       .min_weight_split(1.0)
  //       .min_weight_leaf(1.0)
  //       .fit(&train)?;

  //   let gini_pred_y = gini_model.predict(&test);
  //   let cm = gini_pred_y.confusion_matrix(&test)?;

  //   println!("{:?}", cm);

  //   println!(
  //       "Test accuracy with Gini criterion: {:.2}%",
  //       100.0 * cm.accuracy()
  //   );
  // let tree = DecisionTree::params().fit(&dataset).unwrap();
  // // Get accuracy on training set
  // let accuracy = tree.predict(&dataset).confusion_matrix(&dataset).unwrap().accuracy();

  // assert!(accuracy > 0.9);
}

fn main() {
  testing();
}

#[cfg(test)]
mod test_forest {
  use super::*;

    #[test]
    fn it_works() {
        testing();
    }
}