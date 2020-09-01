use linfa_trees::DecisionTreeParams;


#[derive(Clone, Copy)]
pub enum MaxFeatures {
    // TODO define auto
    Auto,
    Sqrt,
    Log2
}


/// The set of hyperparameters that can be specified for fitting a
/// [decision tree](struct.DecisionTree.html).
#[derive(Clone, Copy)]
pub struct RandomForestParams {
    // single tree parameters
    pub tree_hyperparameters: DecisionTreeParams,
    // number of trees of this ensemble
    pub n_estimators: usize,
    // maximum number of features
    pub max_features: u64,
    // If true, bootstrap samples are used when building trees. If false, the whole dataset is used
    pub bootstrap: bool,
    /*
    pub max_leaf_nodes: Option<u32>,
    pub n_classes: u64,
    pub max_depth: Option<u64>,
    pub min_samples_split: u64,
    pub min_samples_leaf: u64,
    pub min_impurity_decrease: f64,
    */
}

/// A helper struct to build the hyperparameters for a decision tree.
pub struct RandomForestParamsBuilder {
    tree_hyperparameters: DecisionTreeParams,
    n_estimators: usize,
    max_features: Option<MaxFeatures>,
    bootstrap: Option<bool>
}

impl RandomForestParamsBuilder {
    pub fn new(
        tree_hyperparameters: DecisionTreeParams,
        n_estimators: usize,
    ) -> RandomForestParamsBuilder {

        RandomForestParamsBuilder {
            tree_hyperparameters,
            n_estimators,
            max_features: Some(MaxFeatures::Auto),
            bootstrap: Some(false)
        }
    }

    pub fn set_tree_hyperparameters(mut self, tree_hyperparameters: DecisionTreeParams) -> Self {
        self.tree_hyperparameters = tree_hyperparameters;
        self
    }

    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    pub fn max_features(mut self, n_feats: Option<MaxFeatures>) -> Self {
        self.max_features = n_feats;
        self
    }

    pub fn bootstrap(mut self, bootstrap: Option<bool>) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    pub fn build(&self) -> RandomForestParams {
        // TODO set the mandatory fields of RandomForestParams
        let max_features = self.max_features.unwrap_or(MaxFeatures::Auto);

        // TODO
        let max_features = match max_features {
            MaxFeatures::Auto => 42,
            MaxFeatures::Sqrt => 42,
            MaxFeatures::Log2 => 42,
        };
        let bootstrap = self.bootstrap.unwrap_or(false);
        let tree_params = self.tree_hyperparameters;

        RandomForestParams{
            tree_hyperparameters: tree_params,
            n_estimators: self.n_estimators,
            max_features: max_features,
            bootstrap: bootstrap
        }
    }
}

impl RandomForestParams {

}
