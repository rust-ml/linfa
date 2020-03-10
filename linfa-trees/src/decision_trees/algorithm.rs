use crate::decision_trees::hyperparameters::DecisionTreeParams;
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2};
use ndarray_rand::rand::Rng;

pub struct DecisionTree {
    hyperparameters: DecisionTreeParams,
    // tree: TreeNode,
}

impl DecisionTree {
    pub fn fit(
        hyperparameters: DecisionTreeParams,
        x: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
        y: &ArrayBase<impl Data<Elem = u64> + Sync, Ix1>,
        rng: &mut impl Rng,
    ) -> Self {
        Self { hyperparameters }
    }

    pub fn predict(&self, x: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<u64> {
        Array1::ones(5)
    }

    pub fn hyperparameters(&self) -> &DecisionTreeParams {
        &self.hyperparameters
    }
}
