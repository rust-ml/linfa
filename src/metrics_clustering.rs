//! Common metrics for clustering
use crate::dataset::{AsSingleTargets, DatasetBase, Label, Labels, Records};
use crate::error::Result;
use crate::Float;
use ndarray::{ArrayBase, ArrayView1, Data, Ix2};
use std::collections::HashMap;
use std::ops::Sub;

/// Evaluates the quality of a clustering using euclidean distance.
pub trait SilhouetteScore<F> {
    /// Evaluates the quality of a clustering.
    ///
    /// Given a clustered dataset,
    /// the silhouette score for each sample is computed as
    /// the relative difference between the average distance
    /// of the sample to other samples in the same cluster and
    /// the minimum average distance of the sample to samples in
    /// another cluster. This value goes from -1 to +1 when the point
    /// is respectively closer (in average) to points in another cluster and to points in its own cluster.
    ///
    /// Finally, the silhouette score for the clustering is evaluated as the mean
    /// silhouette score of each sample.
    fn silhouette_score(&self) -> Result<F>;
}

struct DistanceCount<F> {
    total_distance: F,
    count: usize,
}

impl<F: Float> DistanceCount<F> {
    /// Sets the total distance from the sample to this cluster to zero
    pub fn reset(&mut self) {
        self.total_distance = F::zero();
    }

    pub fn new(count: usize) -> DistanceCount<F> {
        DistanceCount {
            total_distance: F::zero(),
            count,
        }
    }

    /// Divides the total distance from the sample to this cluster by the number of samples in the cluster
    pub fn mean_distance(&self) -> F {
        self.total_distance / F::cast(self.count)
    }

    /// To be used in the cluster in which the sample is located. The distance from the sample to itself
    /// is zero so it does not get added to the total distance. We can then just divide the total
    /// distance by 1 - #samples in this cluster
    pub fn same_label_mean_distance(&self) -> F {
        if self.count == 1 {
            return F::zero();
        }
        self.total_distance / F::cast(self.count - 1)
    }

    /// adds the distance of `other_sample` from `eval_sample` to the total distance of `eval_sample` from the current cluster
    pub fn add_point(&mut self, eval_sample: ArrayView1<F>, other_sample: ArrayView1<F>) {
        self.total_distance += eval_sample.sub(&other_sample).mapv(|x| x * x).sum().sqrt();
    }
}

impl<
        'a,
        F: Float,
        L: 'a + Label,
        D: Data<Elem = F>,
        T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
    > SilhouetteScore<F> for DatasetBase<ArrayBase<D, Ix2>, T>
{
    fn silhouette_score(&self) -> Result<F> {
        let mut labels: HashMap<L, DistanceCount<F>> = self
            .label_count()
            .remove(0)
            .into_iter()
            .map(|(label, count)| (label, DistanceCount::new(count)))
            .collect();

        // Single label dataset, all points are in the same cluster.
        if labels.len() == 1 {
            return Ok(F::one());
        }

        // Compute and sum silhouette score for each sample
        let score = self
            .sample_iter()
            .map(|sample| {
                // Loops through all samples in the dataset and adds
                // the distance between them and `sample` to the cluster
                // in which they belong

                for other in self.sample_iter() {
                    labels
                        .get_mut(other.1.into_scalar())
                        .unwrap()
                        .add_point(sample.0, other.0);
                }

                // average distance from `sample` to points in its cluster
                let mut a_x = F::zero();
                // minimum average distance from `sample` to another cluster
                // set to none so that it can be initialized as the first value
                let mut b_x: Option<F> = None;

                for (label, counter) in &mut labels {
                    if sample.1.into_scalar() == label {
                        // The cluster of `sample` averages by excluding `sample` from the counting
                        a_x = counter.same_label_mean_distance();
                    } else {
                        // Keep the minimum average distance
                        b_x = match b_x {
                            None => Some(counter.mean_distance()),
                            Some(v) => {
                                if counter.mean_distance() < v {
                                    Some(counter.mean_distance())
                                } else {
                                    Some(v)
                                }
                            }
                        }
                    }
                    counter.reset()
                }
                // Since the single label case was taken care of earlier, here there are at least
                // two clusters so `b_x` can't be `None`
                let b_x = b_x.unwrap();

                // s(x) = (b(x) - a(x)) / max{a(x), b(x)}
                if a_x >= b_x {
                    (b_x - a_x) / a_x
                } else {
                    (b_x - a_x) / b_x
                }
            })
            .sum::<F>();
        let score = score / F::cast(self.records().nsamples());
        Ok(score)
    }
}

#[cfg(test)]
mod tests {
    use crate::metrics_clustering::SilhouetteScore;
    use crate::{Dataset, DatasetBase};
    use approx::assert_abs_diff_eq;
    use ndarray::{concatenate, Array, Array1, Axis, Ix1};

    #[test]
    fn test_silhouette_score() {
        // Two very far apart clusters, each with its own label.
        // This is a very good clustering for silhouette and should return a score very close to +1
        let records = concatenate![
            Axis(0),
            Array::linspace(0f64, 1f64, 10),
            Array::linspace(10000f64, 10001f64, 10)
        ]
        .insert_axis(Axis(1));
        let records = concatenate![Axis(1), records, records];
        let targets = concatenate![Axis(0), Array1::from_elem(10, 0), Array1::from_elem(10, 1)];
        let dataset: Dataset<_, _, Ix1> = (records, targets).into();
        let score = dataset.silhouette_score().unwrap();
        assert_abs_diff_eq!(score, 1f64, epsilon = 1e-3);

        // Two clusters separated into halves very far from each other and each very near an half of the other cluster.
        // Bad but not terrible for silhouette, should return a score slightly negative
        let records = concatenate![
            Axis(0),
            Array::linspace(0f64, 1f64, 5),
            Array::linspace(1f64, 2f64, 5),
            Array::linspace(10000f64, 10001f64, 5),
            Array::linspace(10001f64, 10002f64, 5)
        ]
        .insert_axis(Axis(1));
        let records = concatenate![Axis(1), records, records];
        let targets = concatenate![
            Axis(0),
            Array1::from_elem(5, 0),
            Array1::from_elem(5, 1),
            Array1::from_elem(5, 0),
            Array1::from_elem(5, 1)
        ];
        let dataset: Dataset<_, _, Ix1> = (records, targets).into();
        let score = dataset.silhouette_score().unwrap();
        assert!(score < 0f64);

        // Very bad clustering with a high number of clusters, I expect a very negative value
        let records = Array::linspace(0f64, 10f64, 100).insert_axis(Axis(1));
        let records = concatenate![Axis(1), records, records];
        let targets = Array1::from_shape_fn(100, |i| (i + 3) % 48);
        let dataset: Dataset<_, _, Ix1> = (records, targets).into();
        let score = dataset.silhouette_score().unwrap();
        assert!(score < -0.5f64)
    }

    #[test]
    fn test_empty_labels_as_single_label() {
        let records = Array::linspace(0f64, 1f64, 10).insert_axis(Axis(1));
        let dataset: DatasetBase<_, _> = records.into();
        let score = dataset.silhouette_score().unwrap();
        assert_abs_diff_eq!(score, 1f64, epsilon = 1e-5);
    }
}
