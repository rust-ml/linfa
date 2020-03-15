pub enum SplitQuality {
    Gini,
    Entropy,
}

/// The set of hyperparameters that can be specified for fitting a
/// [decision tree](struct.DecisionTree.html).
pub struct DecisionTreeParams {
    pub n_classes: u64,
    pub split_quality: SplitQuality,
    pub max_depth: Option<u64>,
    pub min_samples_split: u64,
    pub min_samples_leaf: u64,
}

pub struct DecisionTreeParamsBuilder {
    n_classes: u64,
    split_quality: SplitQuality,
    max_depth: Option<u64>,
    min_samples_split: u64,
    min_samples_leaf: u64,
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

    pub fn build(self) -> DecisionTreeParams {
        DecisionTreeParams::build(
            self.n_classes,
            self.split_quality,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
        )
    }
}

impl DecisionTreeParams {
    pub fn new(n_classes: u64) -> DecisionTreeParamsBuilder {
        DecisionTreeParamsBuilder {
            n_classes: n_classes,
            split_quality: SplitQuality::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }

    fn build(
        n_classes: u64,
        split_quality: SplitQuality,
        max_depth: Option<u64>,
        min_samples_split: u64,
        min_samples_leaf: u64,
    ) -> Self {
        // TODO: Check parameters

        DecisionTreeParams {
            n_classes,
            split_quality,
            max_depth,
            min_samples_split,
            min_samples_leaf,
        }
    }
}
