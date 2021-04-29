use std::{cmp::Reverse, collections::BinaryHeap};

use linfa::Float;
use ndarray_stats::DeviationExt;
use ordered_float::NotNan;

use crate::{heap_elem::HeapElem, NearestNeighbour, Point};

fn dist_fn<F: Float>(pt1: &Point<F>, pt2: &Point<F>) -> F {
    pt1.sq_l2_dist(&pt2).unwrap()
}

// Partition the points using median value
fn partition<F: Float>(mut points: Vec<Point<F>>) -> (Vec<Point<F>>, Point<F>, Vec<Point<F>>) {
    debug_assert!(points.len() >= 2);

    // Spread of a dimension is measured using range, which is suceptible to skew. It may be better
    // to use STD or variance.
    let max_spread_dim = (0..points[0].len())
        .map(|dim| {
            // Find the range of each dimension
            let it = points.iter().map(|p| NotNan::new(p[dim]).unwrap());
            // May be faster if we can compute min and max with the same iterator, but compiler might
            // have optimized for that
            let max = it.clone().max().unwrap();
            let min = it.min().unwrap();
            (dim, max - min)
        })
        .max_by_key(|&(_, range)| range)
        .unwrap()
        .0;

    let mid = points.len() / 2;
    // Compute median on the chosen dimension in linear time
    let median = order_stat::kth_by(&mut points, mid, |p1, p2| {
        p1[max_spread_dim].partial_cmp(&p2[max_spread_dim]).unwrap()
    })
    .clone();

    let (mut left, mut right): (Vec<_>, Vec<_>) = points
        .into_iter()
        .partition(|pt| pt[max_spread_dim] < median[max_spread_dim]);
    // We can get an empty left partition with degenerate data where all points are equal and
    // gathered in the right partition.  This ensures that the larger partition will always shrink,
    // guaranteeing algorithm termination.
    if left.is_empty() {
        left.push(right.pop().unwrap());
    }
    (left, median, right)
}

#[derive(Debug, PartialEq)]
enum BallTreeInner<'a, F: Float> {
    Leaf(Point<'a, F>),
    // The sphere is a bounding sphere that encompasses this node (both children)
    Branch {
        center: Point<'a, F>,
        radius: F,
        left: Box<BallTreeInner<'a, F>>,
        right: Box<BallTreeInner<'a, F>>,
    },
}

impl<'a, F: Float> BallTreeInner<'a, F> {
    fn new(mut points: Vec<Point<'a, F>>) -> Self {
        if points.is_empty() {
            unreachable!();
        } else if points.len() == 1 {
            BallTreeInner::Leaf(points.pop().unwrap())
        } else {
            let (aps, center, bps) = partition(points);
            debug_assert!(!aps.is_empty() && !bps.is_empty());
            let radius = aps
                .iter()
                .chain(bps.iter())
                .map(|pt| NotNan::new(dist_fn(pt, &center)).unwrap())
                .max()
                .unwrap()
                .into_inner();
            let (a_tree, b_tree) = (BallTreeInner::new(aps), BallTreeInner::new(bps));
            BallTreeInner::Branch {
                center,
                radius,
                left: Box::new(a_tree),
                right: Box::new(b_tree),
            }
        }
    }

    fn distance(&self, p: &Point<F>) -> F {
        match self {
            // The distance to a leaf is the distance to the single point inside of it
            BallTreeInner::Leaf(p0) => dist_fn(p, p0),
            // The distance to a branch is the distance to the edge of the bounding sphere. Can be
            // negative, which is fine because we're only ever comparing this to the max distance.
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
pub struct BallTree<'a, F: Float>(Option<BallTreeInner<'a, F>>);

impl<'a, F: Float> BallTree<'a, F> {
    /// Construct this `BallTree`. Construction is somewhat expensive, so `BallTree`s
    /// are best constructed once and then used repeatedly.
    pub fn new(points: Vec<Point<'a, F>>) -> Self {
        if points.is_empty() {
            BallTree(None)
        } else {
            BallTree(Some(BallTreeInner::new(points)))
        }
    }

    fn nn_helper<'b>(&self, point: Point<'b, F>, k: Option<usize>, max_radius: F) -> Vec<Point<F>> {
        if let Some(root) = &self.0 {
            let mut out = Vec::new();
            let mut queue = BinaryHeap::new();
            queue.push(HeapElem::new(root.distance(&point), root));
            while queue.len() > 0 {
                let HeapElem {
                    dist: Reverse(dist),
                    elem,
                } = queue.pop().unwrap();
                match elem {
                    BallTreeInner::Leaf(p) => {
                        if dist.into_inner() < max_radius
                            && k.map(|k| out.len() <= k).unwrap_or(true)
                        {
                            out.push(p.reborrow());
                        }
                    }
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
        } else {
            Vec::new()
        }
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

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, aview1, stack, Array1, Array2, Axis};

    use super::*;

    fn assert_partition(
        input: Array2<f64>,
        exp_left: Array2<f64>,
        exp_med: Array1<f64>,
        exp_right: Array2<f64>,
    ) {
        let vec = input.genrows().into_iter().collect();
        let (l, mid, r) = partition(vec);
        assert_abs_diff_eq!(stack(Axis(0), &l).unwrap(), exp_left);
        assert_abs_diff_eq!(mid.to_owned(), exp_med);
        assert_abs_diff_eq!(stack(Axis(0), &r).unwrap(), exp_right);
    }

    #[test]
    fn partition_test() {
        // partition 2 elements
        assert_partition(
            arr2(&[[0.0, 1.0], [2.0, 3.0]]),
            arr2(&[[0.0, 1.0]]),
            arr1(&[2.0, 3.0]),
            arr2(&[[2.0, 3.0]]),
        );
        assert_partition(
            arr2(&[[2.0, 3.0], [0.0, 1.0]]),
            arr2(&[[0.0, 1.0]]),
            arr1(&[2.0, 3.0]),
            arr2(&[[2.0, 3.0]]),
        );

        // Partition along the dimension with highest spread
        assert_partition(
            arr2(&[[0.3, 5.0], [4.5, 7.0], [8.1, 1.5]]),
            arr2(&[[0.3, 5.0]]),
            arr1(&[4.5, 7.0]),
            arr2(&[[4.5, 7.0], [8.1, 1.5]]),
        );

        // Degenerate data
        assert_partition(
            arr2(&[[1.4, 4.3], [1.4, 4.3], [1.4, 4.3], [1.4, 4.3]]),
            arr2(&[[1.4, 4.3]]),
            arr1(&[1.4, 4.3]),
            arr2(&[[1.4, 4.3], [1.4, 4.3], [1.4, 4.3]]),
        );
    }

    #[test]
    fn create_balltree() {
        let arr = arr2(&[[1.0, 2.0]]);
        let tree = BallTreeInner::new(arr.genrows().into_iter().collect());
        assert_eq!(tree, BallTreeInner::Leaf(aview1(&[1.0, 2.0])));
        assert_abs_diff_eq!(tree.distance(&aview1(&[1.0, 3.0])), 1.0);

        let arr = arr2(&[[1.0, 2.0], [-8.0, 4.0], [3.0, 3.0]]);
        let tree = BallTreeInner::new(arr.genrows().into_iter().collect());
        assert_eq!(
            tree,
            BallTreeInner::Branch {
                center: aview1(&[1.0, 2.0]),
                radius: 85.0,
                left: Box::new(BallTreeInner::Leaf(aview1(&[-8.0, 4.0]))),
                right: Box::new(BallTreeInner::Branch {
                    center: aview1(&[3.0, 3.0]),
                    radius: 5.0,
                    left: Box::new(BallTreeInner::Leaf(aview1(&[1.0, 2.0]))),
                    right: Box::new(BallTreeInner::Leaf(aview1(&[3.0, 3.0]))),
                }),
            }
        );
        assert_abs_diff_eq!(tree.distance(&aview1(&[6.0, 3.0])), 26.0 - 85.0);
    }
}
