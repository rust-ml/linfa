/// The possible impurity measures for training.
#[derive(Clone, Copy)]
pub enum SplitQuality {
    Gini,
    Entropy,
}

/// The set of hyperparameters that can be specified for fitting a
/// [decision tree](struct.DecisionTree.html).
#[derive(Clone, Copy)]
pub struct DecisionTreeParams {
    pub n_classes: u64,
    pub split_quality: SplitQuality,
    pub max_depth: Option<u64>,
    pub min_samples_split: u64,
    pub min_samples_leaf: u64,
    pub min_impurity_decrease: f64,
}

/// A helper struct to build the hyperparameters for a decision tree.
pub struct DecisionTreeParamsBuilder {
    n_classes: u64,
    split_quality: SplitQuality,
    max_depth: Option<u64>,
    min_samples_split: u64,
    min_samples_leaf: u64,
    min_impurity_decrease: f64,
}

impl DecisionTreeParamsBuilder {
    pub fn n_classes(mut self, n_classes: u64) -> Self {
        self.n_classes = n_classes;
        self
    }

    pub fn split_quality(mut self, split_quality: SplitQuality) -> Self {
        self.split_quality = split_quality;
        self
    }

    pub fn max_depth(mut self, max_depth: Option<u64>) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn min_samples_split(mut self, min_samples_split: u64) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    pub fn min_samples_leaf(mut self, min_samples_leaf: u64) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    pub fn min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self {
        self.min_impurity_decrease = min_impurity_decrease;
        self
    }

    pub fn build(&self) -> DecisionTreeParams {
        DecisionTreeParams::build(
            self.n_classes,
            self.split_quality,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.min_impurity_decrease,
        )
    }
}

impl DecisionTreeParams {
    /// Defaults are provided if the optional parameters are not specified:
    /// * `split_quality = SplitQuality::Gini`
    /// * `max_depth = None`
    /// * `min_samples_split = 2`
    /// * `min_samples_leaf = 1`
    /// * `min_impurity_decrease = 0.00001`
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn new(n_classes: u64) -> DecisionTreeParamsBuilder {
        DecisionTreeParamsBuilder {
            n_classes,
            split_quality: SplitQuality::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.00001,
        }
    }

    fn build(
        n_classes: u64,
        split_quality: SplitQuality,
        max_depth: Option<u64>,
        min_samples_split: u64,
        min_samples_leaf: u64,
        min_impurity_decrease: f64,
    ) -> Self {
        // TODO: Check parameters

        DecisionTreeParams {
            n_classes,
            split_quality,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
        }
    }
}
