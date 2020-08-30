use std::collections::HashMap;
use std::iter;

use kodama::linkage;
pub use kodama::Method;
use ndarray::{Data, NdFloat};

use linfa_kernel::Kernel;

enum Criterion<T> {
    NumClusters(usize),
    Distance(T),
}

/// Agglomerative Hierarchical clustering
pub struct HierarchicalCluster<T> {
    method: Method,
    stopping: Criterion<T>,
}

impl<T: NdFloat + iter::Sum + Default> HierarchicalCluster<T> {
    /// Select a merging method
    pub fn with_method(mut self, method: Method) -> HierarchicalCluster<T> {
        self.method = method;

        self
    }

    /// Stop merging when a certain number of clusters are reached
    pub fn num_clusters(mut self, num_clusters: usize) -> HierarchicalCluster<T> {
        self.stopping = Criterion::NumClusters(num_clusters);

        self
    }

    /// Stop merging when a certain distance is reached
    pub fn max_distance(mut self, max_distance: T) -> HierarchicalCluster<T> {
        self.stopping = Criterion::Distance(max_distance);

        self
    }

    /// Perform hierarchical clustering for a kernel
    pub fn fit_transform<'a, D: Data<Elem = T>>(self, kernel: &'a Kernel<'a, T, D>) -> Vec<usize> {
        // ignore all similarities below this value
        let threshold = T::from(1e-6).unwrap();

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
        // counter for new clusters, which are formed as unions
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
        tmp
    }
}

impl<T> Default for HierarchicalCluster<T> {
    fn default() -> HierarchicalCluster<T> {
        HierarchicalCluster {
            method: Method::Average,
            stopping: Criterion::NumClusters(2),
        }
    }
}

#[cfg(test)]
mod tests {
    use linfa_kernel::Kernel;
    use ndarray::{Array, Axis};
    use ndarray_rand::{rand_distr::Normal, RandomExt};

    use super::HierarchicalCluster;

    #[test]
    fn test_gaussian_distribution() {
        // we have 10 observations per cluster
        let npoints = 10;
        // generate data
        let entries = ndarray::stack(
            Axis(0),
            &[
                Array::random((npoints, 2), Normal::new(-1., 0.1).unwrap()).view(),
                Array::random((npoints, 2), Normal::new(1., 0.1).unwrap()).view(),
            ],
        )
        .unwrap();

        // apply RBF kernel with sigma=5.0
        let kernel = Kernel::gaussian(&entries, 5.0);

        // perform hierarchical clustering until distance 0.1 is reached
        let ids = HierarchicalCluster::default()
            .max_distance(0.1)
            .fit_transform(&kernel);

        // check that all assigned ids are equal for the first cluster
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
        let ids = HierarchicalCluster::new()
            .num_clusters(2)
            .fit_transform(&kernel);

        // check that all assigned ids are equal for the first cluster
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
}
