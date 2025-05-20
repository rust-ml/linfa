//! Random Forest Classifier
//!
//! An ensemble of decision trees trained on bootstrapped, feature‐subsampled slices of the data.

use linfa::prelude::*;
use linfa::{error::Error, Float, ParamGuard};
use ndarray::{Array1, Array2, Axis};
use rand::{rngs::StdRng, seq::index::sample, Rng, SeedableRng};
use std::marker::PhantomData;

use super::algorithm::DecisionTree;

#[derive(Debug, Clone)]
pub struct RandomForestClassifier<F: Float> {
    trees: Vec<DecisionTree<F, usize>>,
    feature_indices: Vec<Vec<usize>>,
    _phantom: PhantomData<F>,
}

#[derive(Debug, Clone)]
pub struct RandomForestParams<F: Float> {
    inner: RandomForestValidParams<F>,
}

#[derive(Debug, Clone)]
pub struct RandomForestValidParams<F: Float> {
    pub n_trees: usize,
    pub max_depth: Option<usize>,
    pub feature_subsample: f32,
    pub seed: u64,
    _phantom: PhantomData<F>,
}

impl<F: Float> RandomForestParams<F> {
    pub fn new(n_trees: usize) -> Self {
        Self {
            inner: RandomForestValidParams {
                n_trees,
                max_depth: None,
                feature_subsample: 1.0,
                seed: 42,
                _phantom: PhantomData,
            },
        }
    }
    pub fn max_depth(mut self, depth: Option<usize>) -> Self {
        self.inner.max_depth = depth; self
    }
    pub fn feature_subsample(mut self, ratio: f32) -> Self {
        self.inner.feature_subsample = ratio; self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.inner.seed = seed; self
    }
}

impl<F: Float> ParamGuard for RandomForestParams<F> {
    type Checked = RandomForestValidParams<F>;
    type Error = Error;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.inner.n_trees == 0 {
            return Err(Error::Parameters("n_trees must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.inner.feature_subsample) {
            return Err(Error::Parameters(
                "feature_subsample must be in [0, 1]".into(),
            ));
        }
        Ok(&self.inner)
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.inner)
    }
}

/// Bootstrap rows with replacement.
fn bootstrap<F: Float>(
    dataset: &DatasetBase<Array2<F>, Array1<usize>>,
    rng: &mut impl Rng,
) -> DatasetBase<Array2<F>, Array1<usize>> {
    let n = dataset.nsamples();
    let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
    let rec = dataset.records.select(Axis(0), &indices);
    let tgt = dataset.targets.select(Axis(0), &indices);
    Dataset::new(rec, tgt)
}

impl<F: Float + Send + Sync> Fit<Array2<F>, Array1<usize>, Error>
    for RandomForestValidParams<F>
{
    type Object = RandomForestClassifier<F>;

    fn fit(
        &self,
        dataset: &DatasetBase<Array2<F>, Array1<usize>>,
    ) -> Result<Self::Object, Error> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut trees = Vec::with_capacity(self.n_trees);
        let mut feats_list = Vec::with_capacity(self.n_trees);

        let n_features = dataset.records.ncols();
        let n_sub = ((n_features as f32) * self.feature_subsample).ceil() as usize;

        for _ in 0..self.n_trees {
            // 1) bootstrap rows
            let sample_set = bootstrap(dataset, &mut rng);

            // 2) choose feature subset
            let feats = sample(&mut rng, n_features, n_sub)
                .into_iter()
                .collect::<Vec<_>>();
            feats_list.push(feats.clone());

            // 3) train on those features
            let sub_rec = sample_set.records.select(Axis(1), &feats);
            let sub_ds = Dataset::new(sub_rec, sample_set.targets.clone());
            let tree = DecisionTree::params()
                .max_depth(self.max_depth)
                .fit(&sub_ds)?;
            trees.push(tree);
        }

        Ok(RandomForestClassifier {
            trees,
            feature_indices: feats_list,
            _phantom: PhantomData,
        })
    }
}

impl<F: Float> Predict<Array2<F>, Array1<usize>>
    for RandomForestClassifier<F>
{
    fn predict(&self, x: Array2<F>) -> Array1<usize> {
        let n = x.nrows();
        let mut votes = vec![vec![0; n]; 100];

        // For each tree, use its own feature slice:
        for (tree, feats) in self.trees.iter().zip(&self.feature_indices) {
            let sub_x = x.select(Axis(1), feats);
            let preds: Array1<usize> = tree.predict(&sub_x);
            for (i, &c) in preds.iter().enumerate() {
                votes[c][i] += 1;
            }
        }

        Array1::from(
            (0..n)
                .map(|i| {
                    votes.iter()
                         .enumerate()
                         .max_by_key(|(_, vs)| vs[i])
                         .map(|(lbl, _)| lbl)
                         .unwrap_or(0)
                })
                .collect::<Vec<_>>(),
        )
    }
}
