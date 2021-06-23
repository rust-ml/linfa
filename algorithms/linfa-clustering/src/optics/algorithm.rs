use crate::optics::hyperparameters::{OpticsHyperParams, OpticsHyperParamsBuilder};
use float_ord::FloatOrd;
use hnsw::{Params, Searcher, HNSW};
use linfa::traits::Transformer;
use linfa::Float;
use ndarray::{ArrayBase, ArrayView, ArrayView1, Axis, Data, Ix1, Ix2};
use ndarray_stats::DeviationExt;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use space::MetricPoint;
use std::cmp::{max, Ordering};
use std::collections::HashSet;

/// Implementation of euclidean distance for ndarray
struct Euclidean<'a, F>(ArrayView1<'a, F>);

impl<F: Float> MetricPoint for Euclidean<'_, F> {
    fn distance(&self, rhs: &Self) -> u32 {
        let val = self.0.l2_dist(&rhs.0).unwrap();
        space::f64_metric(val)
    }
}

#[derive(Clone, Debug, PartialEq)]
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
/// queries it typically has a higher computational cost that other more specific algorithms.
pub struct Optics;

#[derive(Clone, Debug)]
// This is a struct as in future may want to implement methods on it to get certain metrics from
// the optics cluster distances.
pub struct OpticsAnalysis<'a, F: Float> {
    /// A list of the samples in the dataset sorted and with their reachability and core distances
    /// computed.
    pub orderings: Vec<Sample<'a, F>>,
}

#[derive(Clone, Copy, Debug)]
pub struct Sample<'a, F: Float> {
    /// Index of the sample in the dataset.
    pub index: usize,
    /// A reference to the datum.
    pub observation: ArrayView<'a, F, Ix1>,
    /// The reachability distance of a sample is the distance between the point and it's cluster
    /// core or another point whichever is larger.
    pub reachability_distance: Option<f64>,
    /// The distance to the nth closest point where n is the minimum points to form a cluster.
    pub core_distance: Option<f64>,
}

#[derive(Clone)]
struct Neighbor<'a, F: Float> {
    /// Index of the observation in the dataset
    index: usize,
    /// The observation
    observation: ArrayView<'a, F, Ix1>,
    /// Whether this point has been processed
    processed: bool,
    /// The core distance
    c_distance: Option<FloatOrd<f64>>,
    /// The reachability distance
    r_distance: Option<FloatOrd<f64>>,
}

impl<'a, F: Float> Neighbor<'a, F> {
    /// Create a new neighbor
    fn new(index: usize, observation: ArrayView<'a, F, Ix1>) -> Self {
        Self {
            index,
            observation,
            processed: false,
            c_distance: None,
            r_distance: None,
        }
    }

    /// Set the core distance given the minimum points in a cluster and the points neighbors
    fn set_core_distance(&mut self, min_pts: usize, neighbors: &[Neighbor<'a, F>]) {
        self.c_distance = neighbors
            .get(min_pts - 1)
            .map(|x| FloatOrd(self.observation.l2_dist(&x.observation).unwrap()));
    }

    /// Convert the neighbor to a sample for the user
    fn sample(&self) -> Sample<'a, F> {
        Sample {
            index: self.index,
            observation: self.observation.clone(),
            reachability_distance: self.r_distance.map(|x| x.0),
            core_distance: self.c_distance.map(|x| x.0),
        }
    }
}

impl<'a, F: Float> Eq for Neighbor<'a, F> {}

impl<'a, F: Float> PartialEq for Neighbor<'a, F> {
    fn eq(&self, other: &Self) -> bool {
        self.r_distance == other.r_distance
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl<'a, F: Float> PartialOrd for Neighbor<'a, F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.r_distance.partial_cmp(&other.r_distance)
    }
}

impl<'a, F: Float> Ord for Neighbor<'a, F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.r_distance.cmp(&other.r_distance)
    }
}

impl Optics {
    pub fn params(min_points: usize) -> OpticsHyperParamsBuilder {
        OpticsHyperParams::new(min_points)
    }
}

impl<'a, F: Float, D: Data<Elem = F>> Transformer<&'a ArrayBase<D, Ix2>, OpticsAnalysis<'a, F>>
    for OpticsHyperParams
{
    fn transform(&self, observations: &'a ArrayBase<D, Ix2>) -> OpticsAnalysis<'a, F> {
        let mut result = OpticsAnalysis { orderings: vec![] };

        let mut points = observations
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, x)| Neighbor::new(i, x))
            .collect::<Vec<_>>();

        let mut processed = HashSet::new();
        while processed.len() != points.len() {
            let obs = points
                .iter_mut()
                .enumerate()
                .find(|x| !x.1.processed)
                .unwrap();
            if processed.contains(&obs.0) {
                // We've processed this point so can move on
                continue;
            }
            processed.insert(obs.0);
            let neighbors = find_neighbors(&obs.1.observation, observations, self.tolerance());
            let n = obs.1;
            n.set_core_distance(self.minimum_points(), &neighbors);
            result.orderings.push(n.sample());
            if n.c_distance.is_some() {
                // Here we get a list of "density reachable" samples that haven't been processed
                // and sort them by reachability so we can process the closest ones first.
                let mut seeds = Vec::new();
                get_seeds(n.clone(), &neighbors, &mut points, &processed, &mut seeds);
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
                    let neighbors = find_neighbors(&n.observation, observations, self.tolerance());

                    n.set_core_distance(self.minimum_points(), &neighbors);
                    result.orderings.push(n.sample());
                    if n.c_distance.is_some() {
                        get_seeds(n.clone(), &neighbors, &mut points, &processed, &mut seeds);
                    }
                }
            }
        }
        result
    }
}

/// For a sample find the points which are directly density reachable which have not
/// yet been processed
fn get_seeds<'a, F: Float>(
    sample: Neighbor<'a, F>,
    neighbors: &[Neighbor<'a, F>],
    points: &mut [Neighbor<'a, F>],
    processed: &HashSet<usize>,
    seeds: &mut Vec<usize>,
) {
    for n in neighbors.iter().filter(|x| !processed.contains(&x.index)) {
        let dist = FloatOrd(n.observation.l2_dist(&sample.observation).unwrap());
        let r_dist = max(sample.c_distance.unwrap(), dist);
        match points[n.index].r_distance {
            None => {
                points[n.index].r_distance = Some(r_dist);
                seeds.push(n.index);
            }
            Some(s) if r_dist < s => points[n.index].r_distance = Some(r_dist),
            _ => {}
        }
    }
}

/// Given a candidate point, a list of observations, epsilon and list of already
/// assigned cluster IDs return a list of observations that neighbor the candidate. This function
/// uses euclidean distance and the neighbours are returned in sorted order.
fn find_neighbors<'a, F: Float>(
    candidate: &ArrayBase<impl Data<Elem = F>, Ix1>,
    observations: &'a ArrayBase<impl Data<Elem = F>, Ix2>,
    eps: f64,
) -> Vec<Neighbor<'a, F>> {
    let params = Params::new().ef_construction(observations.nrows());
    let mut searcher = Searcher::default();
    let mut hnsw: HNSW<Euclidean<F>> = HNSW::new_params(params);

    // insert all rows as data points into HNSW graph
    for feature in observations.genrows().into_iter() {
        hnsw.insert(Euclidean(feature), &mut searcher);
    }
    let mut neighbours = vec![space::Neighbor::invalid(); observations.nrows()];
    hnsw.nearest(
        &Euclidean(candidate.view()),
        observations.nrows(),
        &mut searcher,
        &mut neighbours,
    );
    let eps = space::f64_metric(eps);
    let out_of_bounds = neighbours
        .iter()
        .enumerate()
        .find(|(_, x)| x.distance > eps)
        .map(|(i, _)| i);
    if let Some(i) = out_of_bounds {
        // once shrink_to is stablised can switch to that
        neighbours.resize_with(i, space::Neighbor::invalid);
    }

    neighbours
        .iter()
        .map(|x| {
            let observation = observations.row(x.index);
            let distance = candidate.l2_dist(&observation).unwrap();
            Neighbor {
                index: x.index,
                processed: false,
                observation,
                r_distance: Some(FloatOrd(distance)),
                c_distance: None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn simple_optics() {
        let params = Optics::params(3).build();
        let data = vec![1.0, 2.0, 3.0, 8.0, 8.0, 7.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0];
        let data: Array2<f64> = Array2::from_shape_vec((6, 2), data).unwrap();

        let samples = params.transform(&data);
        panic!("Figure out good test");
    }
}
