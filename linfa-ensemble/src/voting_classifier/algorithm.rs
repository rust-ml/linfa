// use ndarray::Array;
// use ndarray::Axis;
// use ndarray_rand::rand_distr::Uniform;
// use ndarray_rand::RandomExt;
// use std::collections::HashMap;
use ndarray::{array, Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use linfa_predictor::Predictor;

/// if hard, use predicted labels for majority rule voting
/// if soft, use the argmax of the sums of the predicted probabilities
#[derive(Clone)]
pub enum Voting {
    Hard,
    Soft
}

#[derive(Clone)]
pub struct VotingClassifier<P: Predictor> {
    pub estimators: Vec<P>,
    pub voting: Voting,
}

impl<P: Predictor> VotingClassifier<P> {
    /// Create new voting classifier. If voting not provided, choose Voting::Hard
    /// by default
    pub fn new(estimators: Vec<P>, voting: Option<Voting>) -> VotingClassifier<P> {
        // assign user-defined weights or balanced weights
        let vote = match voting {
            Some(v) => v,
            None => Voting::Hard
        };

        VotingClassifier {
            estimators: estimators,
            voting: vote,
        }
    }

    /// This method fits the weights of the ensemble (not the single models)
    // pub fn fit(
    //     &mut self,
    //     x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    //     y: &ArrayBase<impl Data<Elem = u64>, Ix1>,
    // ) -> Vec<f64> {
    //     let lin_reg = LinearRegression::new().with_intercept(false);
    //     let A: Array2<f64> = array![[2., 2.], [4., 3.], [2., 2.]];
    //     let b: Array1<f64> = array![2., 4., 1.];
    //     let model = lin_reg.fit(&A, &b).unwrap();
    //     let weights = model.params();
    //     dbg!("weights: ", &weights);
    //     let some = weights.to_vec();
    //     // let _w = &self.weights;
    //     some
    // }

    /// Call .predict() from each estimator and applies fitted weights to results
    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<u64> {
        let result: Array1<u64> = Array1::from(vec![1, 2, 3]);
        // TODO collect predictions from each estimator
        for e in &self.estimators {
            // let single_pred = e.predict(&x);
        }
        // TODO apply voting (hard or soft)

        // TODO return aggregated prediction
        Array1::from(result)
    }


}

#[cfg(test)]
mod tests {
    use linfa_trees::{DecisionTree, DecisionTreeParams};
    use ndarray::Array;

    use super::*;


    #[test]
    fn test_voting_classifier_fit() {
        // Load data
        let data = vec![
            0.54439407, 0.26408166, 0.97446289, 0.81338034, 0.08248497, 0.30045893, 0.35535142,
            0.26975284, 0.46910295, 0.72357513, 0.77458868, 0.09104661, 0.17291617, 0.50215056,
            0.26381918, 0.06778572, 0.92139866, 0.30618514, 0.36123106, 0.90650849, 0.88988489,
            0.44992222, 0.95507872, 0.52735043, 0.42282919, 0.98382015, 0.68076762, 0.4890352,
            0.88607302, 0.24732972, 0.98936691, 0.73508201, 0.16745694, 0.25099697, 0.32681078,
            0.37070237, 0.87316842, 0.85858922, 0.55702507, 0.06624119, 0.3272859, 0.46670468,
            0.87466706, 0.51465624, 0.69996642, 0.04334688, 0.6785262, 0.80599445, 0.6690343,
            0.29780375,
        ];

        let xtrain = Array::from(data).into_shape((10, 5)).unwrap();
        let ytrain = Array1::from(vec![0, 1, 0, 1, 1, 0, 1, 0, 1, 1]);

        let tree_params = DecisionTreeParams::new(2)
            .max_depth(Some(3))
            .min_samples_leaf(2 as u64)
            .build();

        let tree_1 = DecisionTree::fit(tree_params, &xtrain, &ytrain);
        let tree_2 = DecisionTree::fit(tree_params, &xtrain, &ytrain);
        let tree_3 = DecisionTree::fit(tree_params, &xtrain, &ytrain);
        let mut vc = VotingClassifier::new(
                vec![tree_1, tree_2, tree_3],
                None);
        // let some = vc.fit(&xtrain, &ytrain);
        // dbg!("somee: ", &some);

        // let preds = vc.predict(&xtrain, &ytrain);
    }
}
