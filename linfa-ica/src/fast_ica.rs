//! Fast algorithm for Independent Component Analysis (ICA)
//!
//! ICA separates mutivariate signals into their additive, independent subcomponents.
//! ICA is primarily used for separating superimposed signals and not for dimensionality
//! reduction.
//!
//! Input data is whitened (remove underlying correlation) before modeling.

use linfa::{
    dataset::{Dataset, Targets},
    traits::*,
    Float,
};
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::{eigh::Eigh, lapack::UPLO, svd::SVD, Lapack};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;
use rand_isaac::Isaac64Rng;

use crate::error::{FastIcaError, Result};

/// Fast Independent Component Analysis (ICA)
#[derive(Debug)]
pub struct FastIca<F: Float> {
    ncomponents: Option<usize>,
    gfunc: GFunc,
    max_iter: usize,
    tol: F,
    random_state: Option<usize>,
}

impl<F: Float> Default for FastIca<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> FastIca<F> {
    /// Create new FastICA algorithm with default values for its parameters
    pub fn new() -> Self {
        FastIca {
            ncomponents: None,
            gfunc: GFunc::Logcosh(1.),
            max_iter: 200,
            tol: F::from(1e-4).unwrap(),
            random_state: None,
        }
    }

    /// Set the number of components to use, if not set all are used
    pub fn ncomponents(mut self, ncomponents: usize) -> Self {
        self.ncomponents = Some(ncomponents);
        self
    }

    /// G function used in the approximation to neg-entropy, refer [`GFunc`]
    pub fn gfunc(mut self, gfunc: GFunc) -> Self {
        self.gfunc = gfunc;
        self
    }

    /// Set maximum number of iterations during fit
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance on upate at each iteration
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set seed for random number generator for reproducible results.
    pub fn random_state(mut self, random_state: usize) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl<'a, F: Float + Lapack, D: Data<Elem = F>, T: Targets> Fit<'a, ArrayBase<D, Ix2>, T>
    for FastIca<F>
{
    type Object = Result<FittedFastIca<F>>;

    /// Fit the model
    ///
    /// # Errors
    ///
    /// If the [`FastIca::ncomponents`] is set to a number greater than the minimum of
    /// the number of rows and columns
    ///
    /// If the `alpha` value set for [`GFunc::Logcosh`] is not between 1 and 2
    /// inclusive
    fn fit(&self, dataset: &Dataset<ArrayBase<D, Ix2>, T>) -> Result<FittedFastIca<F>> {
        let x = &dataset.records;
        let (nsamples, nfeatures) = (x.nrows(), x.ncols());

        // If the number of components is not set, we take the minimum of
        // the number of rows and columns
        let ncomponents = self.ncomponents.unwrap_or_else(|| nsamples.min(nfeatures));

        // The number of components cannot be greater than the minimum of
        // the number of rows and columns
        if ncomponents > nsamples.min(nfeatures) {
            return Err(FastIcaError::InvalidValue(format!(
                "ncomponents cannot be greater than the min({}, {}), got {}",
                nsamples, nfeatures, ncomponents
            )));
        }

        // We center the input by subtracting the mean of its features
        let xmean = x.mean_axis(Axis(0)).unwrap();
        let mut xcentered = x - &xmean.view().insert_axis(Axis(0));

        // We transpose the centered matrix
        xcentered = xcentered.reversed_axes();

        // We whiten the matrix to remove any potential correlation between
        // the components
        let k = match xcentered.svd(true, false)? {
            (Some(u), s, _) => {
                let s = s.mapv(|x| F::from(x).unwrap());
                (u.slice(s![.., ..nsamples.min(nfeatures)]).to_owned() / s)
                    .t()
                    .slice(s![..ncomponents, ..])
                    .to_owned()
            }
            _ => return Err(FastIcaError::SvdDecomposition),
        };
        let mut xwhitened = k.dot(&xcentered);

        // We multiply the matrix with root of the number of records
        let nsamples_sqrt = F::from((nsamples as f64).sqrt()).unwrap();
        xwhitened.mapv_inplace(|x| x * nsamples_sqrt);

        // We initialize the de-mixing matrix with a uniform distribution
        let w: Array2<f64>;
        if let Some(seed) = self.random_state {
            let mut rng = Isaac64Rng::seed_from_u64(seed as u64);
            w = Array::random_using((ncomponents, ncomponents), Uniform::new(0., 1.), &mut rng);
        } else {
            w = Array::random((ncomponents, ncomponents), Uniform::new(0., 1.));
        }
        let mut w = w.mapv(|x| F::from(x).unwrap());

        // We find the optimized de-mixing matrix
        w = self.ica_parallel(&xwhitened, &w)?;

        // We whiten the de-mixing matrix
        let components = w.dot(&k);

        Ok(FittedFastIca {
            mean: xmean,
            components,
        })
    }
}

impl<F: Float + Lapack> FastIca<F> {
    // Parallel FastICA, Optimization step
    fn ica_parallel(&self, x: &Array2<F>, w: &Array2<F>) -> Result<Array2<F>> {
        let mut w = Self::sym_decorrelation(&w)?;

        let p = x.ncols() as f64;

        for _ in 0..self.max_iter {
            let (gwtx, g_wtx) = self.gfunc.exec(&w.dot(x))?;

            let lhs = gwtx.dot(&x.t()).mapv(|x| x / F::from(p).unwrap());
            let rhs = &w * &g_wtx.insert_axis(Axis(1));
            let wnew = Self::sym_decorrelation(&(lhs - rhs))?;

            // `lim` let us check for convergence between the old and
            // new weight values, we want their dot-product to almost equal one
            let lim = *wnew
                .outer_iter()
                .zip(w.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect::<Array1<F>>()
                .mapv(num_traits::Float::abs)
                .mapv(|x| x - F::from(1.).unwrap())
                .mapv(num_traits::Float::abs)
                .max()
                .unwrap();

            w = wnew;

            if lim < F::from(self.tol).unwrap() {
                break;
            }
        }

        Ok(w)
    }

    // Symmetric decorrelation
    //
    // W <- (W * W.T)^{-1/2} * W
    fn sym_decorrelation(w: &Array2<F>) -> Result<Array2<F>> {
        let (eig_val, eig_vec) = w.dot(&w.t()).eigh(UPLO::Upper)?;
        let eig_val = eig_val.mapv(|x| F::from(x).unwrap());

        let tmp = &eig_vec
            * &(eig_val.mapv(num_traits::Float::sqrt).mapv(|x| {
                // We lower bound the float value at 1e-7 when taking the reciprocal
                let lower_bound = F::from(1e-7).unwrap();
                if x < lower_bound {
                    return num_traits::Float::recip(lower_bound);
                }
                num_traits::Float::recip(x)
            }))
            .insert_axis(Axis(0));

        Ok(tmp.dot(&eig_vec.t()).dot(w))
    }
}

/// Fitted FastICA model for recovering the sources
#[derive(Debug)]
pub struct FittedFastIca<F> {
    mean: Array1<F>,
    components: Array2<F>,
}

impl<F: Float> Predict<&Array2<F>, Array2<F>> for FittedFastIca<F> {
    /// Recover the sources
    fn predict(&self, x: &Array2<F>) -> Array2<F> {
        let xcentered = x - &self.mean.view().insert_axis(Axis(0));
        xcentered.dot(&self.components.t())
    }
}

/// Some standard non-linear functions
#[derive(Debug)]
pub enum GFunc {
    Logcosh(f64),
    Exp,
    Cube,
}

impl GFunc {
    // Function to select the correct non-linear function and execute it
    // returning a tuple, consisting of the first and second derivatives of the
    // non-linear function
    fn exec<A: Float>(&self, x: &Array2<A>) -> Result<(Array2<A>, Array1<A>)> {
        match self {
            Self::Cube => Ok(Self::cube(x)),
            Self::Exp => Ok(Self::exp(x)),
            Self::Logcosh(alpha) => Self::logcosh(x, *alpha),
        }
    }

    fn cube<A: Float>(x: &Array2<A>) -> (Array2<A>, Array1<A>) {
        (
            x.mapv(|x| x.powi(3)),
            x.mapv(|x| A::from(3.).unwrap() * x.powi(2))
                .mean_axis(Axis(1))
                .unwrap(),
        )
    }

    fn exp<A: Float>(x: &Array2<A>) -> (Array2<A>, Array1<A>) {
        let exp = x.mapv(|x| -x.powi(2) / A::from(2.).unwrap());
        (
            x * &exp,
            (x.mapv(|x| A::from(1.).unwrap() - x.powi(2)) * &exp)
                .mean_axis(Axis(1))
                .unwrap(),
        )
    }

    //#[allow(clippy::manual_range_contains)]
    fn logcosh<A: Float>(x: &Array2<A>, alpha: f64) -> Result<(Array2<A>, Array1<A>)> {
        //if alpha < 1.0 || alpha > 2.0 {
        if !(1.0..=2.0).contains(&alpha) {
            return Err(FastIcaError::InvalidValue(format!(
                "alpha must be between 1 and 2 inclusive, got {}",
                alpha
            )));
        }
        let alpha = A::from(alpha).unwrap();

        let gx = x.mapv(|x| (x * alpha).tanh());
        let g_x = gx.mapv(|x| alpha * (A::from(1.).unwrap() - x.powi(2)));

        Ok((gx, g_x.mean_axis(Axis(1)).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    extern crate openblas_src;

    use super::*;
    use linfa::traits::{Fit, Predict};

    use ndarray_rand::rand_distr::StudentT;

    // Test to make sure the number of components set cannot be greater
    // that the minimum of the number of rows and columns of the input
    #[test]
    fn test_ncomponents_err() {
        let input = Dataset::from(Array::random((4, 4), Uniform::new(0.0, 1.0)));
        let ica = FastIca::new().ncomponents(100);
        let ica = ica.fit(&input);
        assert!(ica.is_err());
    }

    // Test to make sure the alpha value of the `GFunc::Logcosh` is between
    // 1 and 2 inclusive
    #[test]
    fn test_logcosh_alpha_err() {
        let input = Dataset::from(Array::random((4, 4), Uniform::new(0.0, 1.0)));
        let ica = FastIca::new().gfunc(GFunc::Logcosh(10.));
        let ica = ica.fit(&input);
        assert!(ica.is_err());
    }

    // Helper macro that produces test-cases with the pattern test_fast_ica_*
    macro_rules! fast_ica_tests {
        ($($name:ident: $gfunc:expr,)*) => {
            paste::item! {
                $(
                    #[test]
                    fn [<test_fast_ica_$name>]() {
                        test_fast_ica($gfunc);
                    }
                )*
            }
        }
    }

    // Tests to make sure all of the `GFunc`'s non-linear functions and the
    // model itself performs well
    fast_ica_tests! {
        exp: GFunc::Exp, cube: GFunc::Cube, logcosh: GFunc::Logcosh(1.0),
    }

    // Helper function that mixes two signal sources sends it to FastICA
    // and makes sure the model can demix them with considerable amount of
    // accuracy
    fn test_fast_ica(gfunc: GFunc) {
        let nsamples = 1000;

        // Center the data and make it have unit variance
        let center_and_norm = |s: &mut Array2<f64>| {
            let mean = s.mean_axis(Axis(0)).unwrap();
            *s -= &mean.insert_axis(Axis(0));
            let std = s.std_axis(Axis(0), 0.);
            *s /= &std.insert_axis(Axis(0));
        };

        // Creaing a sawtooth signal
        let mut source1 = Array::linspace(0., 100., nsamples);
        source1.mapv_inplace(|x| {
            let tmp = 2. * f64::sin(x);
            if tmp > 0. {
                return 0.;
            }
            -1.
        });

        // Creating noise using Student T distribution
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let source2 = Array::random_using((nsamples, 1), StudentT::new(1.0).unwrap(), &mut rng);

        // Column stacking both the sources
        let mut sources = stack![Axis(1), source1.insert_axis(Axis(1)), source2];
        center_and_norm(&mut sources);

        // Mixing the two sources
        let phi: f64 = 0.6;
        let mixing = array![[phi.cos(), phi.sin()], [phi.sin(), -phi.cos()]];
        sources = mixing.dot(&sources.t());
        center_and_norm(&mut sources);

        sources = sources.reversed_axes();

        // We fit and transform using the model to unmix the two sources
        let ica = FastIca::new().ncomponents(2).gfunc(gfunc).random_state(42);

        let sources_dataset = Dataset::from(sources.view());
        let ica = ica.fit(&sources_dataset).unwrap();
        let mut output = ica.predict(&sources);

        center_and_norm(&mut output);

        // Making sure the model output has the right shape
        assert_eq!(output.shape(), &[1000, 2]);

        // The order of the sources in the ICA output is not deterministic,
        // so we account for that here
        let s1 = sources.column(0);
        let s2 = sources.column(1);
        let mut s1_ = output.column(0);
        let mut s2_ = output.column(1);
        if s1_.dot(&s2).abs() > s1_.dot(&s1).abs() {
            s1_ = output.column(1);
            s2_ = output.column(0);
        }

        let similarity1 = s1.dot(&s1_).abs() / (nsamples as f64);
        let similarity2 = s2.dot(&s2_).abs() / (nsamples as f64);

        // We make sure the saw tooth signal identified by ICA using the mixed
        // source is similar to the original sawtooth signal
        // We ignore the noise signal's similarity measure
        assert!(similarity1.max(similarity2) > 0.9);
    }
}
