use linfa::Float;
use ndarray::{Array2, ArrayBase, ArrayView, Axis, Data, Dimension, Ix2, Zip};
use ndarray_stats::DeviationExt;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

/// A distance function that can be used in spatial algorithms such as nearest neighbour.
pub trait Distance<F: Float>: Clone + Send + Sync + Unpin {
    /// Computes the distance between two points. For most spatial algorithms to work correctly,
    /// **this metric must satisfy the Triangle Inequality.**
    ///
    /// Panics if the points have different dimensions.
    fn distance<D: Dimension>(&self, a: ArrayView<F, D>, b: ArrayView<F, D>) -> F;

    /// A faster version of the distance metric that keeps the order of the distance function. That
    /// is, `dist(a, b) > dist(c, d)` implies `rdist(a, b) > rdist(c, d)`. For most algorithms this
    /// is the same as `distance`. Unlike `distance`, this function does **not** need to satisfy
    /// the Triangle Inequality.
    #[inline]
    fn rdistance<D: Dimension>(&self, a: ArrayView<F, D>, b: ArrayView<F, D>) -> F {
        self.distance(a, b)
    }

    /// Converts the result of `rdistance` to `distance`
    #[inline]
    fn rdist_to_dist(&self, rdist: F) -> F {
        rdist
    }

    /// Converts the result of `distance` to `rdistance`
    #[inline]
    fn dist_to_rdist(&self, dist: F) -> F {
        dist
    }
}

/// L1 or [Manhattan](https://en.wikipedia.org/wiki/Taxicab_geometry) distance
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1Dist;
impl<F: Float> Distance<F> for L1Dist {
    #[inline]
    fn distance<D: Dimension>(&self, a: ArrayView<F, D>, b: ArrayView<F, D>) -> F {
        a.l1_dist(&b).unwrap()
    }
}

/// L2 or [Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance) distance
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L2Dist;
impl<F: Float> Distance<F> for L2Dist {
    #[inline]
    fn distance<D: Dimension>(&self, a: ArrayView<F, D>, b: ArrayView<F, D>) -> F {
        F::from(a.l2_dist(&b).unwrap()).unwrap()
    }

    #[inline]
    fn rdistance<D: Dimension>(&self, a: ArrayView<F, D>, b: ArrayView<F, D>) -> F {
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

/// L-infinte or [Chebyshev](https://en.wikipedia.org/wiki/Chebyshev_distance) distance
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LInfDist;
impl<F: Float> Distance<F> for LInfDist {
    #[inline]
    fn distance<D: Dimension>(&self, a: ArrayView<F, D>, b: ArrayView<F, D>) -> F {
        a.linf_dist(&b).unwrap()
    }
}

/// L-p or [Minkowsky](https://en.wikipedia.org/wiki/Minkowski_distance) distance
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct LpDist<F: Float>(pub F);
impl<F: Float> LpDist<F> {
    pub fn new(p: F) -> Self {
        LpDist(p)
    }
}
impl<F: Float> Distance<F> for LpDist<F> {
    #[inline]
    fn distance<D: Dimension>(&self, a: ArrayView<F, D>, b: ArrayView<F, D>) -> F {
        Zip::from(&a)
            .and(&b)
            .fold(F::zero(), |acc, &a, &b| acc + (a - b).abs().powf(self.0))
            .powf(F::one() / self.0)
    }
}

/// [Wasserstein](https://en.wikipedia.org/wiki/Wasserstein_metric) or
/// [Earth Mover's](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) distance.
///
/// The function accepts histograms where each array element is the probability mass at that index.
/// This differs from SciPy's (v1.17.0) `wasserstein_distance` which instead accepts support values and weights,
/// then builds the histograms internally.
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WassersteinDist;
impl<F: Float> Distance<F> for WassersteinDist {
    #[inline]
    fn distance<D: Dimension>(&self, a: ArrayView<F, D>, b: ArrayView<F, D>) -> F {
        let mut cumulative_diff = F::zero();
        let mut dist = F::zero();
        Zip::from(&a).and(&b).for_each(|&a, &b| {
            cumulative_diff += a - b;
            dist += cumulative_diff.abs()
        });
        dist
    }
}

/// Computes a similarity matrix with gaussian kernel and scaling parameter `eps`
///
/// The generated matrix is a upper triangular matrix with dimension NxN (number of observations) and contains the similarity between all permutations of observations
/// similarity
pub fn to_gaussian_similarity<F: Float>(
    observations: &ArrayBase<impl Data<Elem = F>, Ix2>,
    eps: F,
    dist_fn: &impl Distance<F>,
) -> Array2<F> {
    let n_observations = observations.len_of(Axis(0));
    let mut similarity = Array2::eye(n_observations);

    for i in 0..n_observations {
        for j in 0..n_observations {
            let a = observations.row(i);
            let b = observations.row(j);

            let distance = dist_fn.distance(a, b);
            similarity[(i, j)] = (-distance / eps).exp();
        }
    }

    similarity
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;

    use super::*;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<L1Dist>();
        has_autotraits::<L2Dist>();
        has_autotraits::<LInfDist>();
        has_autotraits::<LpDist<f64>>();
        has_autotraits::<WassersteinDist>();
    }

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

    #[test]
    fn wasserstein_dist() {
        dist_test(WassersteinDist, 4.2);
    }

    // The following Wasserstein tests are from SciPy (v1.17.0).
    // However, since SciPy Wasserstein distance has different API as ours,
    // we need to first transform the SciPy parameters into histograms that our API accepts.
    //
    // For example, SciPy values `[1, 3]` with weights `[1, 9]`:
    // At index 1 we have weight 1 out of total weight 10 => 0.1.
    // At index 3 we have weight 9 out of total weight 10 => 0.9.
    // Thus we get a histogram [0.0, 0.1, 0.0, 0.9]

    #[test]
    /// For basic distributions, the value of the Wasserstein distance is straightforward.
    fn wasserstein_simple() {
        let dist = WassersteinDist;

        // SciPy:  u_values=[0, 1], v_values=[0], u_weights=[1, 1], v_weights=[1]
        let u = arr1(&[0.5, 0.5]);
        let v = arr1(&[1.0, 0.0]);
        assert_abs_diff_eq!(dist.distance(u.view(), v.view()), 0.5);

        // SciPy: u_values=[0, 1], v_values=[0], u_weights=[3, 1], v_weights=[1]
        let u = arr1(&[0.75, 0.25]);
        let v = arr1(&[1.0, 0.0]);
        assert_abs_diff_eq!(dist.distance(u.view(), v.view()), 0.25);

        // SciPy: u_values=[0, 2], v_values=[0], u_weights=[1, 1], v_weights=[1]
        let u = arr1(&[0.5, 0.0, 0.5]);
        let v = arr1(&[1.0, 0.0, 0.0]);
        assert_abs_diff_eq!(dist.distance(u.view(), v.view()), 1.0);

        // SciPy: u_values=[0, 1, 2], v_values=[1, 2, 3]
        let u = arr1(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]);
        let v = arr1(&[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert_abs_diff_eq!(dist.distance(u.view(), v.view()), 1.0);
    }

    #[test]
    /// Any distribution moved to itself should have a Wasserstein distance of zero.
    fn wasserstein_same_distribution() {
        let dist = WassersteinDist;

        // SciPy: u_values=[1, 2, 3], v_values=[2, 1, 3]
        let u = arr1(&[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let v = arr1(&[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert_abs_diff_eq!(dist.distance(u.view(), v.view()), 0.0);

        // SciPy: u_values=[1, 1, 1, 4], v_values=[4, 1], u_weights=[1, 1, 1, 1], v_weights=[1, 3]
        let u = arr1(&[0.0, 0.75, 0.0, 0.0, 0.25]);
        let v = arr1(&[0.0, 0.75, 0.0, 0.0, 0.25]);
        assert_abs_diff_eq!(dist.distance(u.view(), v.view()), 0.0);
    }

    #[test]
    /// If the whole distribution is shifted by x, then the Wasserstein distance should be the norm of x.
    fn wasserstein_shift() {
        let dist = WassersteinDist;

        // SciPy: u_values=[0], v_values=[1]
        let u = arr1(&[1.0, 0.0]);
        let v = arr1(&[0.0, 1.0]);
        assert_abs_diff_eq!(dist.distance(u.view(), v.view()), 1.0);

        // SciPy: u_values=[-5], v_values=[5]
        let u = arr1(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_abs_diff_eq!(dist.distance(u.view(), v.view()), 10.0);

        // SciPy: u_values=[1, 2, 3, 4, 5], v_values=[11, 12, 13, 14, 15]
        let u = arr1(&[
            0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let v = arr1(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2,
        ]);
        assert_abs_diff_eq!(dist.distance(u.view(), v.view()), 10.0);
    }

    #[test]
    fn wasserstein_inf_values() {
        let dist = WassersteinDist;

        let u = arr1(&[1.0, f64::INFINITY]);
        let v = arr1(&[1.0, 0.0]);
        assert!(dist.distance(u.view(), v.view()).is_infinite());

        let u = arr1(&[1.0, f64::INFINITY]);
        let v = arr1(&[1.0, f64::INFINITY]);
        assert!(dist.distance(u.view(), v.view()).is_nan());
    }
}
