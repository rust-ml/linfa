use std::{cmp::Reverse, collections::BinaryHeap};

use linfa::Float;
use ndarray::Array1;
use ndarray_stats::DeviationExt;
use ordered_float::NotNan;

use crate::{heap_elem::HeapElem, NearestNeighbour, Point};

fn dist_fn<F: Float>(pt1: &Point<F>, pt2: &Point<F>) -> F {
    pt1.sq_l2_dist(&pt2).unwrap()
}

// Produce a partition of the given points with the following process:
// * Pick a point `a` that is farthest from `points[0]`
// * Pick a point `b` that is farthest from `a`
// * Partition the points into two groups: those closest to `a` and those closest to `b`
//
// This doesn't necessarily form the best partition, since `a` and `b` are not guaranteed
// to be the most distant pair of points, but it's usually sufficient.
fn partition<F: Float>(mut points: Vec<Point<F>>) -> (Vec<Point<F>>, Vec<Point<F>>) {
    assert!(points.len() >= 2);

    let a_i = points
        .iter()
        .enumerate()
        .max_by_key(|(_, a)| NotNan::new(dist_fn(&points[0], a)).unwrap())
        .unwrap()
        .0;

    let b_i = points
        .iter()
        .enumerate()
        .max_by_key(|(_, b)| NotNan::new(dist_fn(&points[a_i], b)).unwrap())
        .unwrap()
        .0;

    let (a_i, b_i) = (a_i.max(b_i), a_i.min(b_i));

    let mut aps = vec![points.swap_remove(a_i)];
    let mut bps = vec![points.swap_remove(b_i)];

    for p in points {
        if dist_fn(&aps[0], &p) < dist_fn(&bps[0], &p) {
            aps.push(p);
        } else {
            bps.push(p);
        }
    }

    (aps, bps)
}

enum BallTreeInner<'a, F: Float> {
    Empty,
    Leaf(Point<'a, F>),
    // The sphere is a bounding sphere that encompasses this node (both children)
    Branch {
        center: Array1<F>,
        radius: F,
        left: Box<BallTreeInner<'a, F>>,
        right: Box<BallTreeInner<'a, F>>,
    },
}

impl<'a, F: Float> BallTreeInner<'a, F> {
    fn new(mut points: Vec<Point<'a, F>>) -> Self {
        if points.is_empty() {
            BallTreeInner::Empty
        } else if points.len() == 1 {
            BallTreeInner::Leaf(points.pop().unwrap())
        } else {
            // TODO lmao
            let (aps, bps) = partition(points);
            let (a_tree, b_tree) = (BallTreeInner::new(aps), BallTreeInner::new(bps));
            BallTreeInner::Branch {
                center: Array1::zeros(0),
                radius: F::zero(),
                left: Box::new(a_tree),
                right: Box::new(b_tree),
            }
        }
    }

    fn distance(&self, p: &Point<F>) -> F {
        match self {
            BallTreeInner::Empty => F::infinity(),
            // The distance to a leaf is the distance to the single point inside of it
            BallTreeInner::Leaf(p0) => dist_fn(p, p0),
            // The distance to a branch is the distance to the edge of the bounding sphere
            BallTreeInner::Branch {
                center,
                radius,
                left: _,
                right: _,
            } => dist_fn(p, &center.view()) - *radius,
        }
    }
}

/// A `BallTree` is a space-partitioning data-structure that allows for finding
/// nearest neighbors in logarithmic time.
///
/// It does this by partitioning data into a series of nested bounding spheres
/// ("balls" in the literature). Spheres are used because it is trivial to
/// compute the distance between a point and a sphere (distance to the sphere's
/// center minus thte radius). The key observation is that a potential neighbor
/// is necessarily closer than all neighbors that are located inside of a
/// bounding sphere that is farther than the aforementioned neighbor.
pub struct BallTree<'a, F: Float>(BallTreeInner<'a, F>);

impl<'a, F: Float> BallTree<'a, F> {
    /// Construct this `BallTree`. Construction is somewhat expensive, so `BallTree`s
    /// are best constructed once and then used repeatedly.
    ///
    /// `panic` if `points.len() != values.len()`
    pub fn new(points: Vec<Point<'a, F>>) -> Self {
        BallTree(BallTreeInner::new(points))
    }

    /// Given a `point`, return an `Iterator` that yields neighbors from closest to
    /// farthest. To get the K nearest neighbors, simply `take` K from the iterator.
    ///
    /// The neighbor, its distance, and associated value is returned.
    fn nn_helper<'b>(&self, point: Point<'b, F>, k: Option<usize>, max_radius: F) -> Vec<Point<F>> {
        let mut out = Vec::new();
        let mut queue = BinaryHeap::new();
        queue.push(HeapElem::new(self.0.distance(&point), &self.0));
        while queue.len() > 0 {
            let HeapElem {
                dist: Reverse(dist),
                elem,
            } = queue.pop().unwrap();
            match elem {
                BallTreeInner::Leaf(p) => {
                    if dist.into_inner() < max_radius && k.map(|k| out.len() <= k).unwrap_or(true) {
                        out.push(p.reborrow());
                    }
                }
                BallTreeInner::Empty => (),
                BallTreeInner::Branch {
                    center: _,
                    radius: _,
                    left,
                    right,
                } => {
                    let dl = left.distance(&point);
                    let dr = right.distance(&point);

                    if dl <= max_radius {
                        queue.push(HeapElem::new(dl, left));
                    }
                    if dr <= max_radius {
                        queue.push(HeapElem::new(dr, right));
                    }
                }
            }
        }
        out
    }
}

impl<'a, F: Float> NearestNeighbour<F> for BallTree<'a, F> {
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Vec<Point<F>> {
        self.nn_helper(point, Some(k), F::infinity())
    }

    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Vec<Point<F>> {
        self.nn_helper(point, None, range)
    }
}
