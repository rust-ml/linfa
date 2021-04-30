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

    #[test]
    fn l1_dist() {
        let a = arr1(&[0.5, 6.6]);
        let b = arr1(&[4.4, 3.0]);
        assert_abs_diff_eq!(
            CommonDistance::L1Dist.distance(a.view(), b.view()),
            7.5,
            epsilon = 1e-3
        );
    }

    #[test]
    fn l2_dist() {
        let a = arr1(&[0.5, 6.6]);
        let b = arr1(&[4.4, 3.0]);
        assert_abs_diff_eq!(
            CommonDistance::L2Dist.distance(a.view(), b.view()),
            5.3075,
            epsilon = 1e-3
        );
    }

    #[test]
    fn sq_l2_dist() {
        let a = arr1(&[0.5, 6.6]);
        let b = arr1(&[4.4, 3.0]);
        assert_abs_diff_eq!(
            CommonDistance::SqL2Dist.distance(a.view(), b.view()),
            28.17,
            epsilon = 1e-3
        );
    }

    #[test]
    fn linf_dist() {
        let a = arr1(&[0.5, 6.6]);
        let b = arr1(&[4.4, 3.0]);
        assert_abs_diff_eq!(
            CommonDistance::LInfDist.distance(a.view(), b.view()),
            3.9,
            epsilon = 1e-3
        );
    }

    #[test]
    fn p_norm() {
        let a = arr1(&[0.5, 6.6]);
        let b = arr1(&[4.4, 3.0]);
        assert_abs_diff_eq!(
            CommonDistance::PNorm(3.3).distance(a.view(), b.view()),
            4.635,
            epsilon = 1e-3
        );
    }
}
