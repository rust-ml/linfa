use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};
// use std::iter::FromIterator;
use linfa_trees::{DecisionTree, DecisionTreeParams};
// use linfa_clustering::generate_blobs;

use crate::random_forest::hyperparameters::{RandomForestParams, RandomForestParamsBuilder, MaxFeatures};


pub struct RandomForest {
    pub hyperparameters: RandomForestParams,
    pub trees: Vec<DecisionTree>,
}

impl RandomForest {

    /// Fit a decision tree using `hyperparamters` on the dataset consisting of
    /// a matrix of features `x` and an array of labels `y`.
    pub fn fit(
        hyperparameters: RandomForestParams,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64>, Ix1>,
    ) -> Self {

        // WIP
        let n_estimators = hyperparameters.n_estimators;
        let mut trees: Vec<DecisionTree> = Vec::with_capacity(n_estimators);
        let single_tree_params = hyperparameters.tree_hyperparameters;

        // WIP check bootstrap
        let _bootstrap = hyperparameters.bootstrap;

        for _ in 0..n_estimators{
            // TODO select a subset of training set with replacement
            let tree = DecisionTree::fit(single_tree_params, &x, &y);
            trees.push(tree);
        }

        // let all_idxs = RowMask::all(x.nrows() as u64);
        // let sorted_indices: Vec<_> = (0..(x.ncols()))
        //     .map(|feature_idx| SortedIndex::of_array_column(&x, feature_idx))
        //     .collect();
        // let root_node = TreeNode::fit(&x, &y, &all_idxs, &hyperparameters, &sorted_indices, 0);

        Self {
            hyperparameters,
            trees,
        }
    }

    /// Make predictions for each row of a matrix of features `x`.
    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<u64> {

        // WIP

        Array1::from(vec![0, 0, 0])
    }

    pub fn hyperparameters(&self) -> &RandomForestParams {
        &self.hyperparameters
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_random_forest_fit() {
        let data = vec![0.54439407, 0.26408166, 0.97446289, 0.81338034, 0.08248497,
                        0.30045893, 0.35535142, 0.26975284, 0.46910295, 0.72357513,
                        0.77458868, 0.09104661, 0.17291617, 0.50215056, 0.26381918,
                        0.06778572, 0.92139866, 0.30618514, 0.36123106, 0.90650849,
                        0.88988489, 0.44992222, 0.95507872, 0.52735043, 0.42282919,
                        0.98382015, 0.68076762, 0.4890352 , 0.88607302, 0.24732972,
                        0.98936691, 0.73508201, 0.16745694, 0.25099697, 0.32681078,
                        0.37070237, 0.87316842, 0.85858922, 0.55702507, 0.06624119,
                        0.3272859 , 0.46670468, 0.87466706, 0.51465624, 0.69996642,
                        0.04334688, 0.6785262 , 0.80599445, 0.6690343 , 0.29780375];

        let xtrain = Array::from(data).into_shape((10, 5)).unwrap();
        dbg!("xtrain: {:?}", &xtrain);

        let y = Array1::from(vec![0, 1, 0, 1, 1, 0, 1, 0, 1, 1]);

        // define single tree parameters from caller
        let tree_params = DecisionTreeParams::new(2).build();
        let dt = DecisionTree::fit(tree_params, &xtrain, &y);
        let preds = dt.predict(&xtrain);
        dbg!("Ground truth: {:?} Predictions: {:?>}", &y, preds);

        // define random forest params from caller
        let rf_params = RandomForestParamsBuilder::new(tree_params, 10)
                    .max_features(Some(MaxFeatures::Auto))
                    .bootstrap(Some(false)).build();

        let rf = RandomForest::fit(rf_params, &xtrain, &y);
        dbg!(&rf.trees[0]);
            // WIP

    }
}
