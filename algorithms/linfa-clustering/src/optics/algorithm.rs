use crate::optics::hyperparams::{OpticsParams, OpticsValidParams};
use linfa::traits::Transformer;
use linfa::Float;
use linfa_nn::distance::{Distance, L2Dist};
use linfa_nn::{CommonNearestNeighbour, NearestNeighbour, NearestNeighbourIndex};
use ndarray::{ArrayView, Ix1, Ix2};
use noisy_float::{checkers::NumChecker, NoisyFloat};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::ops::Index;
use std::slice::SliceIndex;

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
/// OPTICS (Ordering Points To Identify Clustering Structure) is a clustering algorithm that
/// doesn't explicitly cluster the data but instead creates an "augmented ordering" of the dataset
/// representing it's density-based clustering structure. This ordering contains information which
/// is equivalent to the density-based clusterings and can then be used for automatic and
/// interactive cluster analysis.
///
/// OPTICS cluster analysis can be used to derive clusters equivalent to the output of other
/// clustering algorithms such as DBSCAN. However, due to it's more complicated neighborhood
/// queries it typically has a higher computational cost than other more specific algorithms.
///
/// More details on the OPTICS algorithm can be found
/// [here](https://www.wikipedia.org/wiki/OPTICS_algorithm)
pub struct Optics;

/// This struct represents a data point in the dataset with it's associated distances obtained from
/// the OPTICS analysis
#[derive(Debug, Clone)]
pub struct Sample<F> {
    /// Index of the observation in the dataset
    index: usize,
    /// The core distance
    core_distance: Option<F>,
    /// The reachability distance
    reachability_distance: Option<F>,
}

impl<F: Float> Sample<F> {
    /// Create a new neighbor
    fn new(index: usize) -> Self {
        Self {
            index,
            core_distance: None,
            reachability_distance: None,
        }
    }

    /// Index of the sample in the dataset.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The reachability distance of a sample is the distance between the point and it's cluster
    /// core or another point whichever is larger.
    pub fn reachability_distance(&self) -> &Option<F> {
        &self.reachability_distance
    }

    /// The distance to the nth closest point where n is the minimum points to form a cluster.
    pub fn core_distance(&self) -> &Option<F> {
        &self.core_distance
    }
}

impl<F: Float> Eq for Sample<F> {}

impl<F: Float> PartialEq for Sample<F> {
    fn eq(&self, other: &Self) -> bool {
        self.reachability_distance == other.reachability_distance
    }
}

impl<F: Float> PartialOrd for Sample<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.reachability_distance
            .partial_cmp(&other.reachability_distance)
    }
}

impl<F: Float> Ord for Sample<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.reachability_distance
            .map(NoisyFloat::<_, NumChecker>::new)
            .cmp(
                &other
                    .reachability_distance
                    .map(NoisyFloat::<_, NumChecker>::new),
            )
    }
}

/// The analysis from running OPTICS on a dataset, this allows you iterate over the data points and
/// access their core and reachability distances. The ordering of the points also doesn't match
/// that of the dataset instead ordering based on the clustering structure worked out during
/// analysis.
#[derive(Clone, Debug)]
pub struct OpticsAnalysis<F> {
    /// A list of the samples in the dataset sorted and with their reachability and core distances
    /// computed.
    orderings: Vec<Sample<F>>,
}

impl<F: Float> PartialEq for OpticsAnalysis<F> {
    fn eq(&self, other: &Self) -> bool {
        self.orderings == other.orderings
    }
}

impl<F: Float> PartialOrd for OpticsAnalysis<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.orderings.partial_cmp(&other.orderings)
    }
}

impl<F> OpticsAnalysis<F> {
    /// Extracts a slice containing all samples in the dataset
    pub fn as_slice(&self) -> &[Sample<F>] {
        self.orderings.as_slice()
    }

    /// Returns an iterator over the samples in the dataset
    pub fn iter(&self) -> std::slice::Iter<'_, Sample<F>> {
        self.orderings.iter()
    }
}

impl<I, F> Index<I> for OpticsAnalysis<F>
where
    I: SliceIndex<[Sample<F>]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.orderings.index(index)
    }
}

impl Optics {
    /// Configures the hyperparameters with the minimum number of points required to form a cluster
    ///
    /// Defaults are provided if the optional parameters are not specified:
    /// * `tolerance = f64::MAX`
    /// * `dist_fn = L2Dist` (Euclidean distance)
    /// * `nn_algo = KdTree`
    pub fn params<F: Float>(min_points: usize) -> OpticsParams<F, L2Dist, CommonNearestNeighbour> {
        OpticsParams::new(min_points, L2Dist, CommonNearestNeighbour::KdTree)
    }

    /// Configures the hyperparameters with the minimum number of points, a custom distance metric,
    /// and a custom nearest neighbour algorithm
    pub fn params_with<F: Float, D: Distance<F>, N: NearestNeighbour>(
        min_points: usize,
        dist_fn: D,
        nn_algo: N,
    ) -> OpticsParams<F, D, N> {
        OpticsParams::new(min_points, dist_fn, nn_algo)
    }
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour>
    Transformer<ArrayView<'_, F, Ix2>, OpticsAnalysis<F>> for OpticsValidParams<F, D, N>
{
    fn transform(&self, observations: ArrayView<F, Ix2>) -> OpticsAnalysis<F> {
        let mut result = OpticsAnalysis { orderings: vec![] };

        let mut points = (0..observations.nrows())
            .map(Sample::new)
            .collect::<Vec<_>>();

        let nn = match self
            .nn_algo()
            .from_batch(&observations, self.dist_fn().clone())
        {
            Ok(nn) => nn,
            Err(linfa_nn::BuildError::ZeroDimension) => {
                return OpticsAnalysis { orderings: points }
            }
            Err(e) => panic!("Unexpected nearest neighbour error: {}", e),
        };

        // The BTreeSet is used so that the indexes are ordered to make it easy to find next
        // index
        let mut processed = BTreeSet::new();
        let mut index = 0;
        let mut seeds = Vec::new();
        loop {
            if index == points.len() {
                break;
            } else if processed.contains(&index) {
                index += 1;
                continue;
            }
            let mut expected = if processed.is_empty() { 0 } else { index };
            let mut points_index = index;
            // Look for next point to process starting from lowest possible unprocessed index
            for index in processed.range(index..) {
                if expected != *index {
                    points_index = expected;
                    break;
                }
                expected += 1;
            }
            index += 1;
            let neighbors = self.find_neighbors(&*nn, observations.row(points_index));
            let n = &mut points[points_index];
            self.set_core_distance(n, &neighbors, observations);
            if n.core_distance.is_some() {
                seeds.clear();
                // Here we get a list of "density reachable" samples that haven't been processed
                // and sort them by reachability so we can process the closest ones first.
                self.get_seeds(
                    observations,
                    n.clone(),
                    &neighbors,
                    &mut points,
                    &processed,
                    &mut seeds,
                );
                while !seeds.is_empty() {
                    seeds.sort_unstable_by(|a, b| b.cmp(a));
                    let (i, min_point) = seeds
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| points[**a].cmp(&points[**b]))
                        .unwrap();
                    let n = &mut points[*min_point];
                    seeds.remove(i);
                    processed.insert(n.index);
                    let neighbors = self.find_neighbors(&*nn, observations.row(n.index));

                    self.set_core_distance(n, &neighbors, observations);
                    result.orderings.push(n.clone());
                    if n.core_distance.is_some() {
                        self.get_seeds(
                            observations,
                            n.clone(),
                            &neighbors,
                            &mut points,
                            &processed,
                            &mut seeds,
                        );
                    }
                }
            } else {
                // Ensure whole dataset is included so we can see the points with undefined core or
                // reachability distance
                result.orderings.push(n.clone());
                processed.insert(n.index);
            }
        }
        result
    }
}

impl<F: Float, D: Distance<F>, N: NearestNeighbour> OpticsValidParams<F, D, N> {
    /// Given a candidate point, a list of observations, epsilon and list of already
    /// assigned cluster IDs return a list of observations that neighbor the candidate. This function
    /// uses euclidean distance and the neighbours are returned in sorted order.
    fn find_neighbors(
        &self,
        nn: &dyn NearestNeighbourIndex<F>,
        candidate: ArrayView<F, Ix1>,
    ) -> Vec<Sample<F>> {
        // Unwrap here is fine because we don't expect any dimension mismatch when calling
        // within_range with points from the observations
        nn.within_range(candidate, self.tolerance())
            .unwrap()
            .into_iter()
            .map(|(pt, index)| Sample {
                index,
                reachability_distance: Some(self.dist_fn().distance(pt, candidate)),
                core_distance: None,
            })
            .collect()
    }

    /// Set the core distance given the minimum points in a cluster and the points neighbors
    fn set_core_distance(
        &self,
        point: &mut Sample<F>,
        neighbors: &[Sample<F>],
        dataset: ArrayView<F, Ix2>,
    ) {
        let observation = dataset.row(point.index);
        point.core_distance = neighbors
            .get(self.minimum_points() - 1)
            .map(|x| dataset.row(x.index))
            .map(|x| self.dist_fn().distance(observation, x));
    }

    /// For a sample find the points which are directly density reachable which have not
    /// yet been processed
    fn get_seeds(
        &self,
        observations: ArrayView<F, Ix2>,
        sample: Sample<F>,
        neighbors: &[Sample<F>],
        points: &mut [Sample<F>],
        processed: &BTreeSet<usize>,
        seeds: &mut Vec<usize>,
    ) {
        for n in neighbors.iter().filter(|x| !processed.contains(&x.index)) {
            let dist = self
                .dist_fn()
                .distance(observations.row(n.index), observations.row(sample.index));
            let r_dist = F::max(sample.core_distance.unwrap(), dist);
            match points[n.index].reachability_distance {
                None => {
                    points[n.index].reachability_distance = Some(r_dist);
                    seeds.push(n.index);
                }
                Some(s) if r_dist < s => points[n.index].reachability_distance = Some(r_dist),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpticsError;
    use linfa::ParamGuard;
    use linfa_nn::KdTree;
    use ndarray::Array2;
    use std::collections::BTreeSet;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<OpticsAnalysis<f64>>();
        has_autotraits::<Optics>();
        has_autotraits::<Sample<f64>>();
        has_autotraits::<OpticsError>();
        has_autotraits::<OpticsParams<f64, L2Dist, KdTree>>();
        has_autotraits::<OpticsValidParams<f64, L2Dist, KdTree>>();
    }

    #[test]
    fn optics_consistency() {
        let params = Optics::params(3);
        let data = vec![1.0, 2.0, 3.0, 8.0, 8.0, 7.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data).unwrap();

        let samples = params.transform(data.view()).unwrap();

        // Make sure whole dataset is present:
        let indexes = samples
            .orderings
            .iter()
            .map(|x| x.index)
            .collect::<BTreeSet<_>>();
        assert!((0..data.len()).all(|x| indexes.contains(&x)));

        // As we haven't set a tolerance every point should have a core distance
        assert!(samples.orderings.iter().all(|x| x.core_distance.is_some()));
    }

    #[test]
    fn simple_dataset() {
        let params = Optics::params(3).tolerance(4.0);
        //               0    1   2    3     4     5     6     7     8    9     10    11     12
        let data = vec![
            1.0, 2.0, 3.0, 10.0, 18.0, 18.0, 15.0, 2.0, 15.0, 18.0, 3.0, 100.0, 101.0,
        ];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data).unwrap();

        // indexes of groupings of points in the dataset. These will end up with an outlier value
        // in between them to help separate things
        let first_grouping = [0, 1, 2, 7, 10].iter().collect::<BTreeSet<_>>();
        let second_grouping = [4, 5, 6, 8, 9].iter().collect::<BTreeSet<_>>();

        let samples = params.transform(data.view()).unwrap();

        let indexes = samples
            .orderings
            .iter()
            .map(|x| x.index)
            .collect::<BTreeSet<_>>();
        assert!((0..data.len()).all(|x| indexes.contains(&x)));

        assert!(samples
            .orderings
            .iter()
            .take(first_grouping.len())
            .all(|x| first_grouping.contains(&x.index)));
        let skip_len = first_grouping.len() + 1;
        assert!(samples
            .orderings
            .iter()
            .skip(skip_len)
            .take(first_grouping.len())
            .all(|x| second_grouping.contains(&x.index)));

        let anomaly = samples.orderings.iter().find(|x| x.index == 3).unwrap();
        assert!(anomaly.core_distance.is_none());
        assert!(anomaly.reachability_distance.is_none());

        let anomaly = samples.orderings.iter().find(|x| x.index == 11).unwrap();
        assert!(anomaly.core_distance.is_none());
        assert!(anomaly.reachability_distance.is_none());

        let anomaly = samples.orderings.iter().find(|x| x.index == 12).unwrap();
        assert!(anomaly.core_distance.is_none());
        assert!(anomaly.reachability_distance.is_none());
    }

    #[test]
    fn dataset_too_small() {
        let params = Optics::params(4);
        let data = vec![1.0, 2.0, 3.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data).unwrap();

        let samples = params.transform(data.view()).unwrap();

        assert!(samples
            .orderings
            .iter()
            .all(|x| x.core_distance.is_none() && x.reachability_distance.is_none()));
    }

    #[test]
    fn invalid_params() {
        let params = Optics::params(1);
        let data = vec![1.0, 2.0, 3.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data).unwrap();
        assert!(params.transform(data.view()).is_err());

        let params = Optics::params(2);
        assert!(params.transform(data.view()).is_ok());

        let params = params.tolerance(0.0);
        assert!(params.transform(data.view()).is_err());
    }

    #[test]
    fn find_neighbors_test() {
        let data = vec![1.0, 2.0, 10.0, 15.0, 13.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data).unwrap();

        let param = Optics::params(3).tolerance(6.0).check_unwrap();
        let nn = CommonNearestNeighbour::KdTree
            .from_batch(&data, L2Dist)
            .unwrap();

        let neighbors = param.find_neighbors(&*nn, data.row(0));
        assert_eq!(neighbors.len(), 2);
        assert_eq!(
            vec![0, 1],
            neighbors
                .iter()
                .map(|x| x.reachability_distance.unwrap() as u32)
                .collect::<Vec<u32>>()
        );
        assert!(neighbors.iter().all(|x| x.core_distance.is_none()));

        let neighbors = param.find_neighbors(&*nn, data.row(4));
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.iter().all(|x| x.core_distance.is_none()));
        assert_eq!(
            vec![0, 2, 3],
            neighbors
                .iter()
                .map(|x| x.reachability_distance.unwrap() as u32)
                .collect::<Vec<u32>>()
        );
    }

    #[test]
    fn get_seeds_test() {
        let data = vec![1.0, 2.0, 10.0, 15.0, 13.0];
        let data: Array2<f64> = Array2::from_shape_vec((data.len(), 1), data).unwrap();

        let param = Optics::params(3).tolerance(6.0).check_unwrap();
        let nn = CommonNearestNeighbour::KdTree
            .from_batch(&data, L2Dist)
            .unwrap();

        let mut points = (0..data.nrows()).map(Sample::new).collect::<Vec<_>>();

        let neighbors = param.find_neighbors(&*nn, data.row(0));
        // set core distance and make sure it's set correctly given number of neghobrs restriction

        param.set_core_distance(&mut points[0], &neighbors, data.view());
        assert!(points[0].core_distance.is_none());

        let neighbors = param.find_neighbors(&*nn, data.row(4));
        param.set_core_distance(&mut points[4], &neighbors, data.view());
        dbg!(&points);
        assert!(points[4].core_distance.is_some());

        let mut seeds = vec![];
        let mut processed = BTreeSet::new();
        // With a valid core distance make sure neighbours to point are returned in order if
        // unprocessed

        param.get_seeds(
            data.view(),
            points[4].clone(),
            &neighbors,
            &mut points,
            &processed,
            &mut seeds,
        );

        assert_eq!(seeds, vec![4, 3, 2]);

        let mut points = (0..data.nrows()).map(Sample::new).collect::<Vec<_>>();

        // if one of the neighbours has been processed make sure it's not in the seed list

        param.set_core_distance(&mut points[4], &neighbors, data.view());
        processed.insert(3);
        seeds.clear();

        param.get_seeds(
            data.view(),
            points[4].clone(),
            &neighbors,
            &mut points,
            &processed,
            &mut seeds,
        );

        assert_eq!(seeds, vec![4, 2]);

        let mut points = (0..data.nrows()).map(Sample::new).collect::<Vec<_>>();

        // If one of the neighbours has a smaller R distance than it has to the core point make
        // sure it's not added to the seed list

        processed.clear();
        param.set_core_distance(&mut points[4], &neighbors, data.view());
        points[2].reachability_distance = Some(0.001);
        seeds.clear();

        param.get_seeds(
            data.view(),
            points[4].clone(),
            &neighbors,
            &mut points,
            &processed,
            &mut seeds,
        );

        assert_eq!(seeds, vec![4, 3]);
    }
}
