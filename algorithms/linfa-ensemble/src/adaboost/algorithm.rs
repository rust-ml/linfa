use std::{collections::{HashMap}, iter::zip, fs::File};
use std::io::Write;
use linfa_trees::{DecisionTree};
use linfa::{
    dataset::{Labels},
    error::Error,
    error::Result,
    traits::*,
    DatasetBase, Float, Label,
};
use super::Tikz;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use linfa::dataset::AsSingleTargets;
use super::{AdaboostValidParams};
use ndarray::{Array1, ArrayBase, Data, Ix2};
// adaboost will be a vector of stumps

// stump will contain a decision tree and a weight associated with that stump

// dataset will have a row of weights that needs to be updated (passed after every iteration)
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct Stump<F: Float,L: Label> {
    tree: DecisionTree<F,L>,
    weight: f32,
}

impl <F: Float, L: Label + std::fmt::Debug> Stump<F,L> {
    pub fn tree(&self) -> &DecisionTree<F,L> {
        &self.tree
    }
    fn make_stump(tree: DecisionTree<F,L> ,weight: f32) -> Self {
        Stump { tree, weight }
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct Adaboost<F: Float,L: Label> {
    stumps: Vec<Stump<F,L>>,
}

impl<F: Float, L: Label + Default, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array1<L>>
    for Adaboost<F, L>
{
    /// Make predictions for each row of a matrix of features `x`.
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        // Create a vector that has a hashmap with key as label and value as the weight for that label to hold the aggregate of the predictions from every stump for every data record
        let mut map: Vec<HashMap<L,f32>> = Vec::new();
        for stump in self.stumps.iter() {
            // go over each and aggregate the weights of the stump in a hashmap
            for pred in stump.tree.predict(x).iter() {
                let mut map_labels = HashMap::new();
                // if no entry for the label, returns default 0.0
                *map_labels.entry(pred.clone()).or_default() += stump.weight;
                map.push(map_labels);
            }
        }
        // set the label with maximum weight for every record in the target "y"
        for (idx, target) in y.iter_mut().enumerate() {
            // to keep track of the best label and it's weight
            let mut max_entry = None;
            let map_labels = map.get(idx).unwrap();
            // find the max value in map_labels
            for (key, value) in map_labels.iter() {
                // Check if this entry has a greater value than the current maximum
                if max_entry.map_or(true, |(_, max_value)| value > max_value) {
                    // Update the maximum entry
                    max_entry = Some((key, value));
                }
            }
            *target = max_entry.unwrap().0.clone();
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

impl<F: Float, L: Label> Adaboost<F,L>{
    pub fn stumps(&self) -> &Vec<Stump<F,L>> {
        &self.stumps
    }
    pub fn export_to_tikz(&self, mut file: File) {

        file.write_all(Tikz::new(&self).with_legend().to_string().as_bytes(),).unwrap();
        
    }
}

impl<'a, F: Float, L: Label + 'a + std::fmt::Debug, D, T> Fit<ArrayBase<D, Ix2>, T, Error>
    for AdaboostValidParams<F,L>
where
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = Adaboost<F, L>;

    /// Fit an adaboost model using `hyperparamters` on the dataset consisting of
    /// a matrix of features `x` and an array of labels `y`.
    /// https://dept.stat.lsa.umich.edu/~jizhu/pubs/Zhu-SII09.pdf
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        // 1. Assign the weights of each sample in the dataset to be 1/n
        let sample_weight = 1.0 / (dataset.records().nrows() as f32);
        let weights = vec![sample_weight; dataset.records().nrows()];

        // updating the dataset to have the weights by creating a new dataset
        let mut data = DatasetBase::new(dataset.records.view().clone(), dataset.targets.as_targets().clone()).with_feature_names(dataset.feature_names().clone()).with_weights(Array1::from_vec(weights));

        // for lifetime purpose
        let binding = dataset.targets.as_targets();
        // collect all the different unique classes
        let classes: std::collections::HashSet<&L> = binding.iter().collect();
        let num_classes = classes.len();
        
        // lowest f32 value allowed
        let eps = f32::EPSILON;
        let differential = 1.0 - eps;

        let mut stumps: Vec<Stump<F,L>> = Vec::new();
        for i in 0..self.n_estimators() {
            
            let tree_params = self.d_tree_params();
            let tree = tree_params.fit(&data)?;
            // Debug:
            let feats = tree.features();
            println!("Features trained in {i} tree {:?}", feats);

            let mut error = 0.0;

            // predict the data and accumulate the error for wrongly predicted samples
            let predictions = tree.predict(&data);

            for ((idx, pred), weight) in zip(dataset.targets().as_targets().iter().enumerate(), data.weights().unwrap().iter()){
                if predictions[idx] != *pred {
                    error += weight;
                }
            }
            
            // To avoid 0 errors
            error = error.min(differential);

            let alpha: f32 = ((num_classes-1) as f32).ln() + self.learning_rate() * ((1.0 - error) / error ).ln();

            // From sklearn: sample_weight = np.exp(np.log(sample_weight)+ estimator_weight * incorrect * (sample_weight > 0))
            
            // update weights in dataset
            let mut updated_weights: Vec<f32> = Vec::new();
            for ((idx,pred),  weight) in zip(dataset.targets().as_targets().iter().enumerate(), data.weights().unwrap().iter()){
                if *weight > 0.0 && predictions[idx] != *pred {
                    let delta = f32::exp(f32::ln(*weight) + alpha);
                    updated_weights.push(delta);

                } else {
                   updated_weights.push(*weight);
                }
            }
            
            // normalize the weights
            let updated_weights = &Array1::from_vec(updated_weights);
            let normalized_weights = (updated_weights)/(updated_weights.sum());

            // update the weights in the dataset for new stump
            data = DatasetBase::new(dataset.records.view().clone(), dataset.targets.as_targets().clone()).with_feature_names(dataset.feature_names().clone()).with_weights(normalized_weights);

            // push the stump with it's weight
            stumps.push(Stump::make_stump(tree, alpha));
            
        }
        Ok(Adaboost{
            stumps,
        })
        
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    use linfa::{error::Result, Dataset};
    use linfa_trees::DecisionTreeParams;
    use ndarray::{array};

    use crate::AdaboostParams;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<Adaboost<f64, bool>>();
        has_autotraits::<AdaboostValidParams<f32, usize>>();
        has_autotraits::<AdaboostParams<i128, String>>();
    }

    #[test]
    fn small_data() -> Result<()> {
        let data = array![[1., 2., 3.], [1., 2., 4.], [1., 3., 3.5]];
        let targets = array![0, 0, 1];

        let dataset = Dataset::new(data.clone(), targets);
        let model = Adaboost::params().n_estimators(5).d_tree_params(DecisionTreeParams::new().min_weight_leaf(0.00001).min_weight_split(0.00001)).fit(&dataset)?;

        assert_eq!(model.predict(&data), array![0, 0, 1]);

        Ok(())
    }


}