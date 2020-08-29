use std::collections::HashMap;
pub use kodama::Method;
use kodama::linkage;
use linfa_kernel::Kernel;
use ndarray::{Data, NdFloat};
use std::iter;

pub enum Criterion<T> {
    NumClusters(usize),
    Distance(T)
}

pub struct HierarchicalCluster<T> {
    method: Method,
    stopping: Criterion<T>,
}

impl<T: NdFloat + iter::Sum + Default> HierarchicalCluster<T> {
    pub fn new() -> HierarchicalCluster<T> {
        HierarchicalCluster {
            method: Method::Average,
            stopping: Criterion::NumClusters(2)
        }
    }

    pub fn with_method(mut self, method: Method) -> HierarchicalCluster<T> {
        self.method = method;

        self
    }

    pub fn criterion(mut self, criterion: Criterion<T>) -> HierarchicalCluster<T> {
        self.stopping = criterion;

        self
    }

    pub fn fit_transform<'a, D: Data<Elem = T>>(self, kernel: &'a Kernel<'a, T, D>) -> Vec<usize> {
        let mut distance = kernel.to_upper_triangle();
        let num_observations = kernel.size();

        let res = linkage(&mut distance, num_observations, self.method);

        let mut clusters = HashMap::new();
        let mut ct = num_observations;
        for step in res.steps() {
            let should_stop = match self.stopping {
                Criterion::NumClusters(nclusters) => nclusters >= clusters.len(),
                Criterion::Distance(dis) => step.dissimilarity >= dis
            };
            if should_stop {
                break;
            }

            let mut ids = Vec::with_capacity(2);
            if step.cluster1 < num_observations {
                ids.push(step.cluster1);
            } else {
                let mut cl = clusters.remove(&step.cluster1).unwrap();
                ids.append(&mut cl);
            }
            if step.cluster2 < num_observations {
                ids.push(step.cluster2);
            } else {
                let mut cl = clusters.remove(&step.cluster2).unwrap();
                ids.append(&mut cl);
            }
            clusters.insert(ct, ids);
            ct += 1;
        }
        
        let mut tmp = vec![0; num_observations];
        for (i, (_, ids)) in clusters.into_iter().enumerate() {
            for id in ids {
                tmp[id] = i;
            }
        }

        tmp
    }
}

#[cfg(test)]
mod tests {
    use ndarray_rand::{rand_distr::Normal, RandomExt};
    use ndarray::{Array, Axis};
    use linfa_kernel::Kernel;

    use super::{HierarchicalCluster, Criterion};

    #[test]
    fn test_gaussian_distribution() {
        let entries = ndarray::stack(
            Axis(0),
            &[
                Array::random((10, 2), Normal::new(-1., 0.5).unwrap()).view(),
                Array::random((10, 2), Normal::new(0.5, 0.5).unwrap()).view(),
            ],
        )
        .unwrap();

        let kernel = Kernel::linear(&entries);

        let ids = HierarchicalCluster::new()
            .criterion(Criterion::NumClusters(2))
            .fit_transform(&kernel);
        
        dbg!(&ids);
    }
}

