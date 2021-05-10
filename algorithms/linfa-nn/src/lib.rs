//! `linfa-nn` provides Rust implementations of common spatial indexing algorithms, as well as a
//! trait-based interface for performing nearest-neighbour and range queries using these
//! algorithms.
//!
//! ## The big picture
//!
//! `linfa-nn` is a crate in the `linfa` ecosystem, a wider effort to
//! bootstrap a toolkit for classical Machine Learning implemented in pure Rust,
//! kin in spirit to Python's `scikit-learn`.
//!
//! You can find a roadmap (and a selection of good first issues)
//! [here](https://github.com/LukeMathWalker/linfa/issues) - contributors are more than welcome!
//!
//! ## Current state
//!
//! Right now `linfa-nn` provides the following algorithms:
//! * [Linear Scan](struct.LinearSearch.html)
//! * [KD Tree](struct.KdTree.html)
//! * [Ball Tree](struct.BallTree.html)

use distance::Distance;
use linfa::Float;
use ndarray::{Array2, ArrayView1};
use thiserror::Error;

mod balltree;
mod heap_elem;
mod kdtree;
mod linear;

pub mod distance;

pub use crate::{balltree::*, kdtree::*, linear::*};

pub(crate) type Point<'a, F> = ArrayView1<'a, F>;

/// Error returned when building nearest neighbour indices
#[derive(Error, Debug)]
pub enum BuildError {
    #[error("points have dimension of 0")]
    ZeroDimension,
    #[error("leaf size is 0")]
    EmptyLeaf,
}

/// Error returned when performing spatial queries on nearest neighbour indices
#[derive(Error, Debug)]
pub enum NnError {
    #[error("dimensions of query point and stored points are different")]
    WrongDimension,
}

/// Nearest neighbour algorithm builds a spatial index structure out of a batch of points. The
/// distance between points is calculated using a provided distance function. The index implements
/// the [`NearestNeighbourIndex`](trait.NearestNeighbourIndex.html) trait and allows for efficient
/// computing of nearest neighbour and range queries.
///
/// ## Example
///
/// ```rust
/// use rand_isaac::Isaac64Rng;
/// use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
/// use ndarray::{Array1, Array2};
/// use linfa_nn::{distance::*, KdTree, NearestNeighbour};
///
/// // Use seedable RNG for generating points
/// let mut rng = Isaac64Rng::seed_from_u64(40);
/// let n_features = 3;
/// let distr = Uniform::new(-500., 500.);
/// // Randomly generate points for building the index
/// let points = Array2::random_using((5000, n_features), distr, &mut rng);
///
/// // Build a K-D tree with Euclidean distance as the distance function
/// let nn = KdTree::new().from_batch(&points, L2Dist).unwrap();
///
/// let pt = Array1::random_using(n_features, distr, &mut rng);
/// // Compute the 10 nearest points to `pt` in the index
/// let nearest = nn.k_nearest(pt.view(), 10).unwrap();
/// // Compute all points within 100 units of `pt`
/// let range = nn.within_range(pt.view(), 100.0).unwrap();
/// ```
pub trait NearestNeighbour<F: Float, D: Distance<F>> {
    /// Builds a spatial index using a MxN two-dimensional array representing M points with N
    /// dimensions. Also takes `leaf_size`, which specifies the number of elements in the leaf
    /// nodes of tree-like index structures.
    ///
    /// Returns an error if the points have dimensionality of 0 or if the leaf size is 0. If any
    /// value in the batch is NaN or infinite, the behaviour is unspecified.
    fn from_batch_with_leaf_size<'a>(
        &self,
        batch: &'a Array2<F>,
        leaf_size: usize,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbourIndex<F>>, BuildError>;

    /// Builds a spatial index using a default leaf size. See `from_batch_with_leaf_size` for more
    /// information.
    fn from_batch<'a>(
        &self,
        batch: &'a Array2<F>,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbourIndex<F>>, BuildError> {
        self.from_batch_with_leaf_size(batch, 2usize.pow(4), dist_fn)
    }
}

/// A spatial index structure over a set of points, created by `NearestNeighbour`. Allows efficient
/// computation of nearest neighbour and range queries over the set of points. Individual points
/// are represented as one-dimensional array views.
pub trait NearestNeighbourIndex<F: Float> {
    /// Returns the `k` points in the index that are the closest to the provided point. Points are
    /// returned in ascending order of the distance away from the provided points, and less than
    /// `k` points will be returned if the index contains fewer than `k`.
    ///
    /// Returns an error if the provided point has different dimensionality than the index's
    /// points.
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Result<Vec<Point<F>>, NnError>;

    /// Returns all the points in the index that are within the specified distance to the provided
    /// point. The points are not guaranteed to be in any order, though many algorithms return the
    /// points in order of distance.
    ///
    /// Returns an error if the provided point has different dimensionality than the index's
    /// points.
    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Result<Vec<Point<F>>, NnError>;
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, aview1, stack, Axis};
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use ndarray_stats::DeviationExt;
    use noisy_float::{checkers::FiniteChecker, NoisyFloat};
    use rand_isaac::Isaac64Rng;

    use crate::{balltree::BallTree, distance::*, kdtree::KdTree, linear::LinearSearch};

    use super::*;

    fn sort_by_dist<'a>(mut vec: Vec<Point<'a, f64>>, pt: Point<f64>) -> Vec<Point<'a, f64>> {
        vec.sort_by_key(|v| NoisyFloat::<_, FiniteChecker>::new(v.sq_l2_dist(&pt).unwrap()));
        vec
    }

    fn nn_test_empty(builder: &dyn NearestNeighbour<f64, L2Dist>) {
        let points = Array2::zeros((0, 2));
        let nn = builder.from_batch(&points, L2Dist).unwrap();

        let out = nn.k_nearest(aview1(&[0.0, 1.0]), 2).unwrap();
        assert_eq!(out, Vec::<Point<_>>::new());

        let out = nn.k_nearest(aview1(&[4.0, 4.0]), 3).unwrap();
        assert_eq!(out, Vec::<Point<_>>::new());

        let pt = aview1(&[6.0, 3.0]);
        let out = nn.within_range(pt, 9.0).unwrap();
        assert_eq!(out, Vec::<Point<_>>::new());
    }

    fn nn_test_error(builder: &dyn NearestNeighbour<f64, L2Dist>) {
        let points = Array2::zeros((4, 0));
        assert!(builder.from_batch(&points, L2Dist).is_err());

        let points = arr2(&[[0.0, 2.0]]);
        assert!(builder
            .from_batch_with_leaf_size(&points, 0, L2Dist)
            .is_err());
        let nn = builder.from_batch(&points, L2Dist).unwrap();
        assert!(nn.k_nearest(aview1(&[]), 2).is_err());
        assert!(nn.within_range(aview1(&[2.2, 4.4, 5.5]), 4.0).is_err());
    }

    fn nn_test(builder: &dyn NearestNeighbour<f64, L2Dist>, sort_within_range: bool) {
        let points = arr2(&[[0.0, 2.0], [10.0, 4.0], [4.0, 5.0], [7.0, 1.0], [1.0, 7.2]]);
        let nn = builder.from_batch(&points, L2Dist).unwrap();

        let out = nn.k_nearest(aview1(&[0.0, 1.0]), 2).unwrap();
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[0.0, 2.0], [4.0, 5.0]])
        );

        let out = nn.k_nearest(aview1(&[4.0, 4.0]), 3).unwrap();
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[4.0, 5.0], [7.0, 1.0], [1.0, 7.2]])
        );

        let out = nn.k_nearest(aview1(&[4.0, 4.0]), 10).unwrap();
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[4.0, 5.0], [7.0, 1.0], [1.0, 7.2], [0.0, 2.0], [10.0, 4.0]])
        );

        let pt = aview1(&[6.0, 3.0]);
        let mut out = nn.within_range(pt, 4.3).unwrap();
        if sort_within_range {
            out = sort_by_dist(out, pt.clone());
        }
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[7.0, 1.0], [4.0, 5.0], [10.0, 4.0]])
        );
    }

    fn nn_test_degenerate(builder: &dyn NearestNeighbour<f64, L2Dist>) {
        let points = arr2(&[[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]);
        let nn = builder.from_batch(&points, L2Dist).unwrap();

        let out = nn.k_nearest(aview1(&[0.0, 1.0]), 2).unwrap();
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[0.0, 2.0], [0.0, 2.0]])
        );

        let out = nn.k_nearest(aview1(&[4.0, 4.0]), 3).unwrap();
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
        );

        let pt = aview1(&[3.0, 2.0]);
        let out = nn.within_range(pt, 1.0).unwrap();
        assert_eq!(out, Vec::<Point<_>>::new());
        let out = nn.within_range(pt, 20.0).unwrap();
        assert_abs_diff_eq!(stack(Axis(0), &out).unwrap(), points);
    }

    fn nn_test_random<D: 'static + Distance<f64> + Clone>(
        builder: &dyn NearestNeighbour<f64, D>,
        dist_fn: D,
    ) {
        let mut rng = Isaac64Rng::seed_from_u64(40);
        let n_points = 50000;
        let n_features = 3;
        let points =
            Array2::random_using((n_points, n_features), Uniform::new(-50., 50.), &mut rng);

        let linear = LinearSearch::new()
            .from_batch(&points, dist_fn.clone())
            .unwrap();

        let nn = builder.from_batch(&points, dist_fn).unwrap();

        let pt = arr1(&[0., 0., 0.]);
        assert_abs_diff_eq!(
            stack(Axis(0), &nn.k_nearest(pt.view(), 5).unwrap()).unwrap(),
            stack(Axis(0), &linear.k_nearest(pt.view(), 5).unwrap()).unwrap()
        );
        assert_abs_diff_eq!(
            stack(
                Axis(0),
                &sort_by_dist(nn.within_range(pt.view(), 15.0).unwrap(), pt.view())
            )
            .unwrap(),
            stack(
                Axis(0),
                &sort_by_dist(linear.within_range(pt.view(), 15.0).unwrap(), pt.view())
            )
            .unwrap()
        );

        let pt = arr1(&[-3.4, 10., 0.95]);
        assert_abs_diff_eq!(
            stack(Axis(0), &nn.k_nearest(pt.view(), 30).unwrap()).unwrap(),
            stack(Axis(0), &linear.k_nearest(pt.view(), 30).unwrap()).unwrap()
        );
        assert_abs_diff_eq!(
            stack(
                Axis(0),
                &sort_by_dist(nn.within_range(pt.view(), 25.0).unwrap(), pt.view())
            )
            .unwrap(),
            stack(
                Axis(0),
                &sort_by_dist(linear.within_range(pt.view(), 25.0).unwrap(), pt.view())
            )
            .unwrap()
        );
    }

    macro_rules! nn_tests {
        ($mod:ident, $builder:ident, $sort:expr $(, $_u:ident)?) => {
            mod $mod {
                use super::*;

                #[test]
                fn empty() {
                    nn_test_empty(&$builder::default());
                }

                #[test]
                fn error() {
                    nn_test_error(&$builder::default());
                }

                #[test]
                fn normal() {
                    nn_test(&$builder::default(), $sort);
                }

                #[test]
                fn degenerate() {
                    nn_test_degenerate(&$builder::default());
                }

                $(
                    #[test]
                    fn random_l2() {
                        let $_u: () = ();
                        nn_test_random(&$builder::default(), L2Dist);
                    }

                    #[test]
                    fn random_l1() {
                        nn_test_random(&$builder::default(), L1Dist);
                    }
                )?
            }
        };
    }

    nn_tests!(linear_search, LinearSearch, true);
    nn_tests!(kdtree, KdTree, false, _u);
    nn_tests!(balltree, BallTree, false, _u);
}
