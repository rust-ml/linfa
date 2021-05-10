use linfa::Float;
use ndarray::Zip;
use ndarray_stats::DeviationExt;

use crate::Point;

// Should satisfy triangle inequality (no squared Euclidean)
pub trait Distance<F: Float> {
    // Panics if a and b are not of equal dimension
    fn distance(&self, a: Point<F>, b: Point<F>) -> F;

    // Fast distance metric that keeps the order of the distance function
    #[inline]
    fn rdistance(&self, a: Point<F>, b: Point<F>) -> F {
        self.distance(a, b)
    }

    #[inline]
    fn rdist_to_dist(&self, rdist: F) -> F {
        rdist
    }

    #[inline]
    fn dist_to_rdist(&self, dist: F) -> F {
        dist
    }
}

#[derive(Debug, Clone)]
pub struct L1Dist;
impl<F: Float> Distance<F> for L1Dist {
    #[inline]
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        a.l1_dist(&b).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct L2Dist;
impl<F: Float> Distance<F> for L2Dist {
    #[inline]
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        F::from(a.l2_dist(&b).unwrap()).unwrap()
    }

    #[inline]
    fn rdistance(&self, a: Point<F>, b: Point<F>) -> F {
        F::from(a.sq_l2_dist(&b).unwrap()).unwrap()
    }

    #[inline]
    fn rdist_to_dist(&self, rdist: F) -> F {
        rdist.sqrt()
    }

    #[inline]
    fn dist_to_rdist(&self, dist: F) -> F {
        dist.powi(2)
    }
}

#[derive(Debug, Clone)]
pub struct LInfDist;
impl<F: Float> Distance<F> for LInfDist {
    #[inline]
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        a.linf_dist(&b).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct LpDist<F: Float>(F);
impl<F: Float> Distance<F> for LpDist<F> {
    #[inline]
    fn distance(&self, a: Point<F>, b: Point<F>) -> F {
        Zip::from(&a)
            .and(&b)
            .fold(F::zero(), |acc, &a, &b| acc + (a - b).abs().powf(self.0))
            .powf(F::one() / self.0)
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;

    use super::*;

    fn dist_test<D: Distance<f64>>(dist: D, result: f64) {
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
        dist_test(L1Dist, 7.5);
    }

    #[test]
    fn l2_dist() {
        dist_test(L2Dist, 5.3075);

        // Check squared distance
        let a = arr1(&[0.5, 6.6]);
        let b = arr1(&[4.4, 3.0]);
        assert_abs_diff_eq!(L2Dist.rdistance(a.view(), b.view()), 28.17, epsilon = 1e-3);
    }

    #[test]
    fn linf_dist() {
        dist_test(LInfDist, 3.9);
    }

    #[test]
    fn lp_dist() {
        dist_test(LpDist(3.3), 4.635);
    }
}
