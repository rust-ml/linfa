use linfa::Float;
use ndarray::Zip;
use ndarray_stats::DeviationExt;

use crate::Point;

// Should satisfy triangle inequality (no squared Euclidean)
pub trait Distance<F: Float> {
    // Panics if a and b are not of equal dimension
    fn distance(&self, a: Point<F>, b: Point<F>) -> F;

    // Fast distance metric that keeps the order of the distance function
    fn rdistance(&self, a: Point<F>, b: Point<F>) -> F {
        self.distance(a, b)
    }

    fn rdist_to_dist(&self, rdist: F) -> F {
        rdist
    }

    fn dist_to_rdist(&self, dist: F) -> F {
        dist
    }
}

pub struct L1Dist;
impl<F: Float> Distance<F> for L1Dist {
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        a.l1_dist(&b).unwrap()
    }
}

pub struct L2Dist;
impl<F: Float> Distance<F> for L2Dist {
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        F::from(a.l2_dist(&b).unwrap()).unwrap()
    }

    fn rdistance(&self, a: Point<F>, b: Point<F>) -> F {
        F::from(a.sq_l2_dist(&b).unwrap()).unwrap()
    }

    fn rdist_to_dist(&self, rdist: F) -> F {
        rdist.sqrt()
    }

    fn dist_to_rdist(&self, dist: F) -> F {
        dist.powi(2)
    }
}

pub struct LInfDist;
impl<F: Float> Distance<F> for LInfDist {
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        a.linf_dist(&b).unwrap()
    }
}

pub struct LpDist<F: Float>(F);
impl<F: Float> Distance<F> for LpDist<F> {
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        Zip::from(&a)
            .and(&b)
            .fold(F::zero(), |acc, &a, &b| acc + (a - b).abs().powf(self.0))
            .powf(F::one() / self.0)
    }
}

#[non_exhaustive]
pub enum CommonDistance<F> {
    /// Manhattan distance
    L1Dist,
    /// Euclidean distance
    L2Dist,
    /// Chebyshev distance
    LInfDist,
    /// Minkowski distance
    LpDist(F),
}

impl<F: Float> Distance<F> for CommonDistance<F> {
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        match self {
            Self::L1Dist => L1Dist.distance(a, b),
            Self::L2Dist => L2Dist.distance(a, b),
            Self::LInfDist => LInfDist.distance(a, b),
            Self::LpDist(p) => LpDist(*p).distance(a, b),
        }
    }

    fn rdistance(&self, a: Point<F>, b: Point<F>) -> F {
        match self {
            Self::L1Dist => L1Dist.rdistance(a, b),
            Self::L2Dist => L2Dist.rdistance(a, b),
            Self::LInfDist => LInfDist.rdistance(a, b),
            Self::LpDist(p) => LpDist(*p).rdistance(a, b),
        }
    }

    fn rdist_to_dist(&self, rdist: F) -> F {
        match self {
            Self::L1Dist => L1Dist.rdist_to_dist(rdist),
            Self::L2Dist => L2Dist.rdist_to_dist(rdist),
            Self::LInfDist => LInfDist.rdist_to_dist(rdist),
            Self::LpDist(p) => LpDist(*p).rdist_to_dist(rdist),
        }
    }

    fn dist_to_rdist(&self, dist: F) -> F {
        match self {
            Self::L1Dist => L1Dist.dist_to_rdist(dist),
            Self::L2Dist => L2Dist.dist_to_rdist(dist),
            Self::LInfDist => LInfDist.dist_to_rdist(dist),
            Self::LpDist(p) => LpDist(*p).dist_to_rdist(dist),
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
        let ab = dist.distance(a.view(), b.view());
        assert_abs_diff_eq!(ab, result, epsilon = 1e-3);
        assert_abs_diff_eq!(dist.rdist_to_dist(dist.dist_to_rdist(ab)), ab);

        let a = arr1(&[f64::INFINITY, 6.6]);
        let b = arr1(&[4.4, f64::NEG_INFINITY]);
        assert!(dist.distance(a.view(), b.view()).is_infinite());

        // Triangle equality
        let a = arr1(&[0.5, 6.6]);
        let b = arr1(&[4.4, 3.0]);
        let c = arr1(&[-4.5, 3.3]);
        let ab = dist.distance(a.view(), b.view());
        let bc = dist.distance(b.view(), c.view());
        let ac = dist.distance(a.view(), c.view());
        assert!(ab + bc > ac)
    }

    #[test]
    fn l1_dist() {
        dist_test(CommonDistance::L1Dist, 7.5);
    }

    #[test]
    fn l2_dist() {
        dist_test(CommonDistance::L2Dist, 5.3075);

        // Check squared distance
        let a = arr1(&[0.5, 6.6]);
        let b = arr1(&[4.4, 3.0]);
        assert_abs_diff_eq!(
            CommonDistance::L2Dist.rdistance(a.view(), b.view()),
            28.17,
            epsilon = 1e-3
        );
    }

    #[test]
    fn linf_dist() {
        dist_test(CommonDistance::LInfDist, 3.9);
    }

    #[test]
    fn lp_dist() {
        dist_test(CommonDistance::LpDist(3.3), 4.635);
    }
}
