use linfa_trees::DecisionTree;
use crate::random_forest::hyperparameters::RandomForestParams;
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};
use ndarray_rand::rand_distr::Uniform;
use ndarray::Axis;
use ndarray::Array;
use ndarray_rand::RandomExt;
use std::collections::HashMap;


pub struct RandomForest {
    pub hyperparameters: RandomForestParams,
    pub trees: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn fit(
        hyperparameters: RandomForestParams,
        x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64>, Ix1>,
        max_n_rows: Option<usize>
    ) -> Self {
        let n_estimators = hyperparameters.n_estimators;
        let mut trees: Vec<DecisionTree> = Vec::with_capacity(n_estimators);
        let single_tree_params = hyperparameters.tree_hyperparameters;
        let max_n_rows = max_n_rows.unwrap_or(x.nrows());

        //TODO check bootstrap
        let _bootstrap = hyperparameters.bootstrap;
        for _ in 0..n_estimators {
            // Bagging here
            let rnd_idx = Array::random((1, max_n_rows), Uniform::new(0, x.nrows())).into_raw_vec();
            let xsample = x.select(Axis(0), &rnd_idx);
            let ysample = y.select(Axis(0), &rnd_idx);
            let tree = DecisionTree::fit(single_tree_params, &xsample, &ysample);
            trees.push(tree);
    }

    Self {
        hyperparameters: hyperparameters,
        trees: trees
        }
    }

    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<u64> {
        let ntrees = self.hyperparameters.n_estimators;
        assert!(ntrees > 0, "Run .fit() method first");

        let mut result: Vec<u64> = Vec::with_capacity(x.nrows());
        let flattened: Vec<Vec<u64>> = self.trees.iter()
        .map(|tree| {
            let single_pred = tree.predict(&x).to_vec();
            single_pred
        }).collect();

        for sample_idx in 0..x.nrows() {
            // hashmap to store most common prediction across trees
            let mut counter_stats: HashMap<u64, u64> = HashMap::new();
            for sp in &flattened {
                *counter_stats.entry(sp[sample_idx]).or_insert(0) += 1;
            }

            // aggregate counters to final prediction
            let final_pred = counter_stats
                            .iter()
                            .max_by(|a,b| a.1.cmp(&b.1))
                            .map(|(k, _v)| k)
                            .unwrap();

            result.push(*final_pred);
        }
        Array1::from(result)
    }

    /// Collect features from each tree in the forest and return hashmap(feature_idx: counts)
    ///
    pub fn feature_importances(&self)-> HashMap<usize, usize>{
        let mut counter: HashMap<usize, usize> = HashMap::new();
        for st in &self.trees {
            // features in the single tree
            let st_feats = st.features();
            for f in st_feats.iter() {
                *counter.entry(*f).or_insert(0) += 1
            }
        }

        counter
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use linfa_trees::DecisionTreeParams;
    use crate::random_forest::hyperparameters::{RandomForestParamsBuilder,
            MaxFeatures};

    #[test]
    fn test_random_forest_fit() {
        // Load data
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
        let ytrain = Array1::from(vec![0, 1, 0, 1, 1, 0, 1, 0, 1, 1]);

        // Define parameters of single tree
        let tree_params = DecisionTreeParams::new(2)
                                                .max_depth(Some(3))
                                                .min_samples_leaf(2 as u64)
                                                .build();
        // Define parameters of random forest
        let ntrees = 500;
        let rf_params = RandomForestParamsBuilder::new(tree_params, ntrees)
                                                        .max_features(Some(MaxFeatures::Auto))
                                                        .build();
        let rf = RandomForest::fit(rf_params, &xtrain, &ytrain, None);
        assert_eq!(rf.trees.len(), ntrees);

        let imp = rf.feature_importances();
        dbg!("Feature importances: ", imp);

        let preds = rf.predict(&xtrain);
        dbg!("Predictions: {}", preds);
    }

}