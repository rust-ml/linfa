use linfa_trees::DecisionTreeParams;
use ndarray::{array, Array, ArrayBase, Data, Ix1};
use linfa_predictor::Predictor;

fn main() {
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

    let mod1 = DecisionTree::fit(tree_params, &xtrain, &ytrain);
    let mod2 = DecisionTree::fit(tree_params, &xtrain, &ytrain);
    let mod3 = DecisionTree::fit(tree_params, &xtrain, &ytrain);
    let vc = VotingClassifier::new(vec![mod1, mod2, mod3]);
    let preds = vc.predict(&xtrain).unwrap();
    dbg!("preds: ", &preds);
    assert_eq!(preds.len(), xtrain.shape()[0]);
}
