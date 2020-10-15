use linfa_trees::DecisionTreeParams;

#[derive(Clone, Copy)]
pub struct RandomForestParams {
    // number of decision trees to fit
    pub n_estimators: usize,
    // single tree hyperparameters
    pub tree_hyperparameters: DecisionTreeParams,
    // max number of features for the ensemble
    pub max_features: u64,
}

#[derive(Clone, Copy)]
pub enum MaxFeatures {
    Auto,
    Sqrt,
    Log2,
}

pub struct RandomForestParamsBuilder {
    tree_hyperparameters: DecisionTreeParams,
    n_estimators: usize,
    max_features: Option<MaxFeatures>,
    bootstrap: Option<bool>,
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
            bootstrap: Some(false),
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
        let max_features = self.max_features.unwrap_or(MaxFeatures::Auto);

        let max_features = match max_features {
            MaxFeatures::Auto => 42,
            MaxFeatures::Log2 => 42,
            MaxFeatures::Sqrt => 42,
        };

        RandomForestParams {
            tree_hyperparameters: self.tree_hyperparameters,
            n_estimators: self.n_estimators,
            max_features,
        }
    }
}
