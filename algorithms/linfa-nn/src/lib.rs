use distance::{CommonDistance, Distance};
use linfa::Float;
use ndarray::{Array2, ArrayView1};
use thiserror::Error;

mod heap_elem;

pub mod balltree;
pub mod distance;
pub mod kdtree;
pub mod linear;

pub(crate) type Point<'a, F> = ArrayView1<'a, F>;

#[derive(Error, Debug)]
pub enum BuildError {
    #[error("points have dimension of 0")]
    ZeroDimension,
}

#[derive(Error, Debug)]
pub enum NnError {
    #[error("dimensions of query point and stored points are different")]
    WrongDimension,
}

pub trait NearestNeighbour<F: Float> {
    // Returns nearest in order. Might want wrap in result or return iterator
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Result<Vec<Point<F>>, NnError>;

    // Does not have any particular order, though some algorithms may returns these in order of
    // distance.
    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Result<Vec<Point<F>>, NnError>;
}

pub trait NearestNeighbourBuilder<F: Float, D: Distance<F> = CommonDistance<F>> {
    fn from_batch<'a>(
        &self,
        batch: &'a Array2<F>,
        dist_fn: D,
    ) -> Result<Box<dyn 'a + NearestNeighbour<F>>, BuildError>;
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, aview1, stack, Axis};
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use ndarray_stats::DeviationExt;
    use noisy_float::{checkers::FiniteChecker, NoisyFloat};
    use rand_isaac::Isaac64Rng;

    use crate::{
        balltree::BallTreeBuilder, distance::CommonDistance, kdtree::KdTreeBuilder,
        linear::LinearSearchBuilder,
    };

    use super::*;

    fn sort_by_dist<'a>(mut vec: Vec<Point<'a, f64>>, pt: Point<f64>) -> Vec<Point<'a, f64>> {
        vec.sort_by_key(|v| NoisyFloat::<_, FiniteChecker>::new(v.sq_l2_dist(&pt).unwrap()));
        vec
    }

    fn nn_test_empty(builder: &dyn NearestNeighbourBuilder<f64>) {
        let points = Array2::zeros((0, 2));
        let nn = builder.from_batch(&points, CommonDistance::L2Dist).unwrap();

        let out = nn.k_nearest(aview1(&[0.0, 1.0]), 2).unwrap();
        assert_eq!(out, Vec::<Point<_>>::new());

        let out = nn.k_nearest(aview1(&[4.0, 4.0]), 3).unwrap();
        assert_eq!(out, Vec::<Point<_>>::new());

        let pt = aview1(&[6.0, 3.0]);
        let out = nn.within_range(pt, 9.0).unwrap();
        assert_eq!(out, Vec::<Point<_>>::new());
    }

    fn nn_test_error(builder: &dyn NearestNeighbourBuilder<f64>) {
        let points = Array2::zeros((4, 0));
        assert!(builder.from_batch(&points, CommonDistance::L2Dist).is_err());

        let points = arr2(&[[0.0, 2.0]]);
        let nn = builder.from_batch(&points, CommonDistance::L2Dist).unwrap();
        assert!(nn.k_nearest(aview1(&[]), 2).is_err());
        assert!(nn.within_range(aview1(&[2.2, 4.4, 5.5]), 4.0).is_err());
    }

    fn nn_test(builder: &dyn NearestNeighbourBuilder<f64>, sort_within_range: bool) {
        let points = arr2(&[[0.0, 2.0], [10.0, 4.0], [4.0, 5.0], [7.0, 1.0], [1.0, 7.2]]);
        let nn = builder.from_batch(&points, CommonDistance::L2Dist).unwrap();

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

    fn nn_test_degenerate(builder: &dyn NearestNeighbourBuilder<f64>) {
        let points = arr2(&[[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]);
        let nn = builder.from_batch(&points, CommonDistance::L2Dist).unwrap();

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

    fn nn_test_random(builder: &dyn NearestNeighbourBuilder<f64>, dist_fn: CommonDistance<f64>) {
        let mut rng = Isaac64Rng::seed_from_u64(40);
        let n_points = 50000;
        let n_features = 3;
        let points =
            Array2::random_using((n_points, n_features), Uniform::new(-50., 50.), &mut rng);

        let linear = LinearSearchBuilder::new()
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
        ($mod:ident, $builder:ident, $sort:expr) => {
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

                #[test]
                fn random_l2() {
                    nn_test_random(&$builder::default(), CommonDistance::L2Dist);
                }

                #[test]
                fn random_l1() {
                    nn_test_random(&$builder::default(), CommonDistance::L1Dist);
                }
            }
        };
    }

    nn_tests!(linear_search, LinearSearchBuilder, true);
    nn_tests!(kdtree, KdTreeBuilder, false);
    nn_tests!(balltree, BallTreeBuilder, false);
}
