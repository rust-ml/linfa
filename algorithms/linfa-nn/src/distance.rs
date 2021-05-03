use linfa::Float;
use ndarray::Zip;
use ndarray_stats::DeviationExt;

use crate::Point;

pub trait Distance<F: Float> {
    // Panics if a and b are not of equal dimension
    fn distance(&self, a: Point<F>, b: Point<F>) -> F;
}

pub enum CommonDistance<F> {
    /// Manhattan distance
    L1Dist,
    /// Euclidean distance
    L2Dist,
    /// Squared Euclidean distance
    SqL2Dist,
    /// Chebyshev distance
    LInfDist,
    /// Minkowski distance
    PNorm(F),
}

impl<F: Float> Distance<F> for CommonDistance<F> {
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        match self {
            Self::L1Dist => a.l1_dist(&b).unwrap(),
            Self::L2Dist => F::from(a.l2_dist(&b).unwrap()).unwrap(),
            Self::SqL2Dist => a.sq_l2_dist(&b).unwrap(),
            Self::LInfDist => a.linf_dist(&b).unwrap(),
            Self::PNorm(p) => Zip::from(&a)
                .and(&b)
                .fold(F::zero(), |acc, &a, &b| acc + (a - b).abs().powf(*p))
                .powf(F::one() / *p),
        }
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;

    use super::*;

    fn dist_test(dist: CommonDistance<f64>, result: f64) {
        let a = arr1(&[0.5, 6.6]);
        let b = arr1(&[4.4, 3.0]);
        assert_abs_diff_eq!(dist.distance(a.view(), b.view()), result, epsilon = 1e-3);

        let a = arr1(&[f64::INFINITY, 6.6]);
        let b = arr1(&[4.4, f64::NEG_INFINITY]);
        assert!(dist.distance(a.view(), b.view()).is_infinite());
    }

    #[test]
    fn l1_dist() {
        dist_test(CommonDistance::L1Dist, 7.5);
    }

    #[test]
    fn l2_dist() {
        dist_test(CommonDistance::L2Dist, 5.3075);
    }

    #[test]
    fn sq_l2_dist() {
        dist_test(CommonDistance::SqL2Dist, 28.17);
    }

    #[test]
    fn linf_dist() {
        dist_test(CommonDistance::LInfDist, 3.9);
    }

    #[test]
    fn p_norm() {
        dist_test(CommonDistance::PNorm(3.3), 4.635);
    }
}
