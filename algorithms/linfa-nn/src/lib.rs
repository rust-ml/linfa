use linfa::Float;
use ndarray::{Array2, ArrayView1};

pub mod balltree;
mod heap_elem;
pub mod kdtree;
pub mod linear;

pub(crate) type Point<'a, F> = ArrayView1<'a, F>;

pub trait NearestNeighbour<F: Float> {
    // Returns nearest in order. Might want wrap in result or return iterator
    fn k_nearest<'b>(&self, point: Point<'b, F>, k: usize) -> Vec<Point<F>>;

    // Does not have any particular order, though some algorithms may returns these in order of
    // distance.
    fn within_range<'b>(&self, point: Point<'b, F>, range: F) -> Vec<Point<F>>;
}

pub trait NearestNeighbourBuilder<F: Float> {
    fn from_batch<'a>(&self, batch: &'a Array2<F>) -> Box<dyn 'a + NearestNeighbour<F>>;
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, aview1, stack, Axis};
    use ndarray_stats::DeviationExt;
    use ordered_float::NotNan;

    use crate::{kdtree::KdTreeBuilder, linear::LinearSearchBuilder};

    use super::*;

    fn nn_test_empty(builder: &dyn NearestNeighbourBuilder<f64>) {
        let points = Array2::zeros((0, 2));
        let nn = builder.from_batch(&points);

        let out = nn.k_nearest(aview1(&[0.0, 1.0]), 2);
        assert_eq!(out, Vec::<Point<_>>::new());

        let out = nn.k_nearest(aview1(&[4.0, 4.0]), 3);
        assert_eq!(out, Vec::<Point<_>>::new());

        let pt = aview1(&[6.0, 3.0]);
        let out = nn.within_range(pt, 9.0);
        assert_eq!(out, Vec::<Point<_>>::new());
    }

    fn nn_test(builder: &dyn NearestNeighbourBuilder<f64>, sort_within_range: bool) {
        let points = arr2(&[[0.0, 2.0], [10.0, 4.0], [4.0, 5.0], [7.0, 1.0], [1.0, 7.2]]);
        let nn = builder.from_batch(&points);

        let out = nn.k_nearest(aview1(&[0.0, 1.0]), 2);
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[0.0, 2.0], [4.0, 5.0]])
        );

        let out = nn.k_nearest(aview1(&[4.0, 4.0]), 3);
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[4.0, 5.0], [7.0, 1.0], [1.0, 7.2]])
        );

        let pt = aview1(&[6.0, 3.0]);
        let mut out = nn.within_range(pt, 9.0);
        if sort_within_range {
            out.sort_by_key(|v| NotNan::new(v.sq_l2_dist(&pt).unwrap()).unwrap());
        }
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[7.0, 1.0], [4.0, 5.0]])
        );
    }

    fn nn_test_degenerate(builder: &dyn NearestNeighbourBuilder<f64>) {
        let points = arr2(&[[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]);
        let nn = builder.from_batch(&points);

        let out = nn.k_nearest(aview1(&[0.0, 1.0]), 2);
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[0.0, 2.0], [0.0, 2.0]])
        );

        let out = nn.k_nearest(aview1(&[4.0, 4.0]), 3);
        assert_abs_diff_eq!(
            stack(Axis(0), &out).unwrap(),
            arr2(&[[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
        );

        let pt = aview1(&[3.0, 2.0]);
        let out = nn.within_range(pt, 1.0);
        assert_eq!(out, Vec::<Point<_>>::new());
        let out = nn.within_range(pt, 20.0);
        assert_abs_diff_eq!(stack(Axis(0), &out).unwrap(), points);
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
                fn normal() {
                    nn_test(&$builder::default(), $sort);
                }

                #[test]
                fn degenerate() {
                    nn_test_degenerate(&$builder::default());
                }
            }
        };
    }

    nn_tests!(linear_search, LinearSearchBuilder, true);
    nn_tests!(kdtree, KdTreeBuilder, false);
}
