//! # Hierarchical Clustering
//!
//! `linfa-hierarchical` provides an implementation of agglomerative hierarchical clustering.
//! In this clustering algorithm, each point is first considered as a separate cluster. During each
//! step, two points are merged into new clusters, until a stopping criterion is reached. The distance
//! between the points is computed as the negative-log transform of the similarity kernel.
//!
//! _Documentation_: [latest](https://docs.rs/linfa-hierarchical).
//!
//! ## The big picture
//!
//! `linfa-hierarchical` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem,
//! a wider effort to bootstrap a toolkit for classical Machine Learning implemented in pure Rust,
//! akin in spirit to Python's `scikit-learn`.
//!
//! ## Current state
//!
//! `linfa-hierarchical` implements agglomerative hierarchical clustering with support of the
//! [kodama](https://docs.rs/kodama/0.2.3/kodama/) crate.

use std::collections::HashMap;

use kodama::linkage;
pub use kodama::Method;

use linfa::param_guard::TransformGuard;
use linfa::traits::Transformer;
use linfa::Float;
use linfa::{dataset::DatasetBase, ParamGuard};
use linfa_kernel::Kernel;

pub use error::{HierarchicalError, Result};

mod error;

/// Criterion when to stop merging
///
/// The criterion defines at which point the merging process should stop. This can be either, when
/// a certain number of clusters is reached, or the distance becomes larger than a maximal
/// distance.
#[derive(Clone, Debug, PartialEq)]
pub enum Criterion<F: Float> {
    NumClusters(usize),
    Distance(F),
}

/// Agglomerative hierarchical clustering
///
/// In this clustering algorithm, each point is first considered as a separate cluster. During each
/// step, two points are merged into new clusters, until a stopping criterion is reached. The distance
/// between the points is computed as the negative-log transform of the similarity kernel.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct HierarchicalCluster<T: Float>(ValidHierarchicalCluster<T>);

/// Checked version of [`HierarchicalCluster`](`HierarchicalCluster`)
#[derive(Clone, Debug, PartialEq)]
pub struct ValidHierarchicalCluster<T: Float> {
    method: Method,
    stopping: Criterion<T>,
}

impl<F: Float> ParamGuard for HierarchicalCluster<F> {
    type Checked = ValidHierarchicalCluster<F>;
    type Error = HierarchicalError<F>;

    fn check_ref(&self) -> std::result::Result<&Self::Checked, Self::Error> {
        match self.0.stopping {
            Criterion::NumClusters(x) if x == 0 => Err(
                HierarchicalError::InvalidStoppingCondition(self.0.stopping.clone()),
            ),
            Criterion::Distance(x) if x.is_negative() || x.is_nan() || x.is_infinite() => Err(
                HierarchicalError::InvalidStoppingCondition(self.0.stopping.clone()),
            ),
            _ => Ok(&self.0),
        }
    }

    fn check(self) -> std::result::Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
impl<F: Float> TransformGuard for HierarchicalCluster<F> {}

impl<F: Float> HierarchicalCluster<F> {
    /// Select a merging method
    pub fn with_method(mut self, method: Method) -> HierarchicalCluster<F> {
        self.0.method = method;
        self
    }

    /// Stop merging when a certain number of clusters are reached
    ///
    /// In the fitting process points are merged until a certain criterion is reached. With this
    /// option the merging process will stop, when the number of clusters drops below this value.
    pub fn num_clusters(mut self, num_clusters: usize) -> HierarchicalCluster<F> {
        self.0.stopping = Criterion::NumClusters(num_clusters);
        self
    }

    /// Stop merging when a certain distance is reached
    ///
    /// In the fitting process points are merged until a certain criterion is reached. With this
    /// option the merging process will stop, then the distance exceeds this value.
    pub fn max_distance(mut self, max_distance: F) -> HierarchicalCluster<F> {
        self.0.stopping = Criterion::Distance(max_distance);
        self
    }
}

impl<F: Float> Transformer<Kernel<F>, DatasetBase<Kernel<F>, Vec<usize>>>
    for ValidHierarchicalCluster<F>
{
    /// Perform hierarchical clustering of a similarity matrix
    ///
    /// Returns the class id for each data point
    fn transform(&self, kernel: Kernel<F>) -> DatasetBase<Kernel<F>, Vec<usize>> {
        // ignore all similarities below this value
        let threshold = F::cast(1e-6);

        // transform similarities to distances with log transformation
        let mut distance = kernel
            .to_upper_triangle()
            .into_iter()
            .map(|x| {
                if x > threshold {
                    -x.ln()
                } else {
                    -threshold.ln()
                }
            })
            .collect::<Vec<_>>();

        // call kodama linkage function
        let num_observations = kernel.size();
        let res = linkage(&mut distance, num_observations, self.method);

        // post-process results, iterate through merging step until threshold is reached
        // at the beginning every node is in its own cluster
        let mut clusters = (0..num_observations)
            .map(|x| (x, vec![x]))
            .collect::<HashMap<_, _>>();

        // counter for new clusters, which are formed as unions of previous ones
        let mut ct = num_observations;

        for step in res.steps() {
            let should_stop = match self.stopping {
                Criterion::NumClusters(max_clusters) => clusters.len() <= max_clusters,
                Criterion::Distance(dis) => step.dissimilarity >= dis,
            };

            // break if one of the two stopping condition is reached
            if should_stop {
                break;
            }

            // combine ids from both clusters
            let mut ids = Vec::with_capacity(2);
            let mut cl = clusters.remove(&step.cluster1).unwrap();
            ids.append(&mut cl);
            let mut cl = clusters.remove(&step.cluster2).unwrap();
            ids.append(&mut cl);

            // insert into hashmap and increase counter
            clusters.insert(ct, ids);
            ct += 1;
        }

        // flatten resulting clusters and reverse index
        let mut tmp = vec![0; num_observations];
        for (i, (_, ids)) in clusters.into_iter().enumerate() {
            for id in ids {
                tmp[id] = i;
            }
        }

        // return node_index -> cluster_index map
        DatasetBase::new(kernel, tmp)
    }
}

impl<F: Float, T> Transformer<DatasetBase<Kernel<F>, T>, DatasetBase<Kernel<F>, Vec<usize>>>
    for ValidHierarchicalCluster<F>
{
    /// Perform hierarchical clustering of a similarity matrix
    ///
    /// Returns the class id for each data point
    fn transform(&self, dataset: DatasetBase<Kernel<F>, T>) -> DatasetBase<Kernel<F>, Vec<usize>> {
        self.transform(dataset.records)
    }
}

/// Initialize hierarchical clustering, which averages in-cluster points and stops when two
/// clusters are reached.
impl<T: Float> Default for ValidHierarchicalCluster<T> {
    fn default() -> Self {
        Self {
            method: Method::Average,
            stopping: Criterion::NumClusters(2),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::HierarchicalError;
    use linfa::traits::Transformer;
    use linfa_kernel::{Kernel, KernelMethod};
    use ndarray::{Array, Axis};
    use ndarray_rand::{rand_distr::Normal, RandomExt};

    use super::{Criterion, HierarchicalCluster, ValidHierarchicalCluster};

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<Criterion<f64>>();
        has_autotraits::<HierarchicalCluster<f64>>();
        has_autotraits::<ValidHierarchicalCluster<f64>>();
        has_autotraits::<HierarchicalError<f64>>();
    }

    #[test]
    fn test_blobs() {
        // we have 10 observations per cluster
        let npoints = 10;
        // generate data
        let entries = ndarray::concatenate(
            Axis(0),
            &[
                Array::random((npoints, 2), Normal::new(-1., 0.1).unwrap()).view(),
                Array::random((npoints, 2), Normal::new(1., 0.1).unwrap()).view(),
            ],
        )
        .unwrap();

        let kernel = Kernel::params()
            .method(KernelMethod::Gaussian(5.0))
            .transform(entries.view());

        let kernel = HierarchicalCluster::default()
            .max_distance(0.1)
            .transform(kernel)
            .unwrap();

        // check that all assigned ids are equal for the first cluster
        let ids = kernel.targets();
        let first_cluster_id = &ids[0];
        assert!(ids
            .iter()
            .take(npoints)
            .all(|item| item == first_cluster_id));

        // and for the second
        let second_cluster_id = &ids[npoints];
        assert!(ids
            .iter()
            .skip(npoints)
            .all(|item| item == second_cluster_id));

        // the cluster ids shouldn't be equal
        assert_ne!(first_cluster_id, second_cluster_id);

        // perform hierarchical clustering until we have two clusters left
        let kernel = HierarchicalCluster::default()
            .num_clusters(2)
            .transform(kernel)
            .unwrap();

        // check that all assigned ids are equal for the first cluster
        let ids = kernel.targets();
        let first_cluster_id = &ids[0];
        assert!(ids
            .iter()
            .take(npoints)
            .all(|item| item == first_cluster_id));

        // and for the second
        let second_cluster_id = &ids[npoints];
        assert!(ids
            .iter()
            .skip(npoints)
            .all(|item| item == second_cluster_id));

        // the cluster ids shouldn't be equal
        assert_ne!(first_cluster_id, second_cluster_id);
    }

    #[test]
    fn test_noise() {
        // generate 1000 normal distributed points
        let data = Array::random((100, 2), Normal::new(0., 1.0).unwrap());

        let kernel = Kernel::params()
            .method(KernelMethod::Linear)
            .transform(data.view());

        let predictions = HierarchicalCluster::default()
            .max_distance(3.0)
            .transform(kernel)
            .unwrap();

        dbg!(&predictions.targets());
    }
}
