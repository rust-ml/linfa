//! Fast algorithm for Independent Component Analysis (ICA)

use linfa::{
    dataset::{DatasetBase, Records, WithLapack, WithoutLapack},
    traits::*,
    Float,
};
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::{eigh::Eigh, solveh::UPLO, svd::SVD};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

use crate::error::{FastIcaError, Result};
use crate::hyperparams::FastIcaValidParams;

impl<F: Float, D: Data<Elem = F>, T> Fit<ArrayBase<D, Ix2>, T, FastIcaError>
    for FastIcaValidParams<F>
{
    type Object = FastIca<F>;

    /// Fit the model
    ///
    /// # Errors
    ///
    /// If the [`FastIca::ncomponents`] is set to a number greater than the minimum of
    /// the number of rows and columns
    ///
    /// If the `alpha` value set for [`GFunc::Logcosh`] is not between 1 and 2
    /// inclusive
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let x = &dataset.records;
        let (nsamples, nfeatures) = (x.nsamples(), x.nfeatures());
        if dataset.nsamples() == 0 {
            return Err(FastIcaError::NotEnoughSamples);
        }

        // If the number of components is not set, we take the minimum of
        // the number of rows and columns
        let ncomponents = self
            .ncomponents()
            .unwrap_or_else(|| nsamples.min(nfeatures));

        // The number of components cannot be greater than the minimum of
        // the number of rows and columns
        if ncomponents > nsamples.min(nfeatures) {
            return Err(FastIcaError::InvalidValue(format!(
                "ncomponents cannot be greater than the min({}, {}), got {}",
                nsamples, nfeatures, ncomponents
            )));
        }

        // We center the input by subtracting the mean of its features
        // safe unwrap because we already returned an error on zero samples
        let xmean = x.mean_axis(Axis(0)).unwrap();
        let mut xcentered = x - &xmean.view().insert_axis(Axis(0));

        // We transpose the centered matrix
        xcentered = xcentered.reversed_axes();

        // We whiten the matrix to remove any potential correlation between
        // the components
        let xcentered = xcentered.with_lapack();
        let k = match xcentered.svd(true, false)? {
            (Some(u), s, _) => {
                let s = s.mapv(F::Lapack::cast);
                (u.slice_move(s![.., ..nsamples.min(nfeatures)]) / s)
                    .t()
                    .slice(s![..ncomponents, ..])
                    .to_owned()
            }
            _ => return Err(FastIcaError::SvdDecomposition),
        };

        let mut xwhitened = k.dot(&xcentered).without_lapack();
        let k = k.without_lapack();

        // We multiply the matrix with root of the number of records
        let nsamples_sqrt = F::cast(nsamples).sqrt();
        xwhitened.mapv_inplace(|x| x * nsamples_sqrt);

        // We initialize the de-mixing matrix with a uniform distribution
        let w: Array2<f64>;
        if let Some(seed) = self.random_state() {
            let mut rng = Xoshiro256Plus::seed_from_u64(*seed as u64);
            w = Array::random_using((ncomponents, ncomponents), Uniform::new(0., 1.), &mut rng);
        } else {
            w = Array::random((ncomponents, ncomponents), Uniform::new(0., 1.));
        }
        let mut w = w.mapv(F::cast);

        // We find the optimized de-mixing matrix
        w = self.ica_parallel(&xwhitened, &w)?;

        // We whiten the de-mixing matrix
        let components = w.dot(&k);

        Ok(FastIca {
            mean: xmean,
            components,
        })
    }
}

impl<F: Float> FastIcaValidParams<F> {
    // Parallel FastICA, Optimization step
    fn ica_parallel(&self, x: &Array2<F>, w: &Array2<F>) -> Result<Array2<F>> {
        let mut w = Self::sym_decorrelation(w)?;

        let p = x.ncols() as f64;

        for _ in 0..self.max_iter() {
            let (gwtx, g_wtx) = self.gfunc().exec(&w.dot(x))?;

            let lhs = gwtx.dot(&x.t()).mapv(|x| x / F::cast(p));
            let rhs = &w * &g_wtx.insert_axis(Axis(1));
            let wnew = Self::sym_decorrelation(&(lhs - rhs))?;

            // `lim` let us check for convergence between the old and
            // new weight values, we want their dot-product to almost equal one
            let lim = *wnew
                .outer_iter()
                .zip(w.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect::<Array1<F>>()
                .mapv(|x| x.abs())
                .mapv(|x| x - F::cast(1.))
                .mapv(|x| x.abs())
                .max()
                .unwrap();

            w = wnew;

            if lim < F::cast(self.tol()) {
                break;
            }
        }

        Ok(w)
    }

    // Symmetric decorrelation
    //
    // W <- (W * W.T)^{-1/2} * W
    fn sym_decorrelation(w: &Array2<F>) -> Result<Array2<F>> {
        let (eig_val, eig_vec) = w.dot(&w.t()).with_lapack().eigh(UPLO::Upper)?;
        let eig_val = eig_val.mapv(F::cast);
        let eig_vec = eig_vec.without_lapack();

        let tmp = &eig_vec
            * &(eig_val.mapv(|x| x.sqrt()).mapv(|x| {
                // We lower bound the float value at 1e-7 when taking the reciprocal
                let lower_bound = F::cast(1e-7);
                if x < lower_bound {
                    return lower_bound.recip();
                }
                x.recip()
            }))
            .insert_axis(Axis(0));

        Ok(tmp.dot(&eig_vec.t()).dot(w))
    }
}

/// Fitted FastICA model for recovering the sources
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct FastIca<F> {
    mean: Array1<F>,
    components: Array2<F>,
}

impl<F: Float> PredictInplace<Array2<F>, Array2<F>> for FastIca<F> {
    /// Recover the sources
    fn predict_inplace(&self, x: &Array2<F>, y: &mut Array2<F>) {
        assert_eq!(
            y.shape(),
            &[x.nrows(), self.components.nrows()],
            "The number of data points must match the number of output targets."
        );

        let xcentered = x - &self.mean.view().insert_axis(Axis(0));
        *y = xcentered.dot(&self.components.t());
    }

    fn default_target(&self, x: &Array2<F>) -> Array2<F> {
        Array2::zeros((x.nrows(), self.components.nrows()))
    }
}

/// Some standard non-linear functions
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
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
            x.mapv(|x| A::cast(3.) * x.powi(2))
                .mean_axis(Axis(1))
                .unwrap(),
        )
    }

    fn exp<A: Float>(x: &Array2<A>) -> (Array2<A>, Array1<A>) {
        let exp = x.mapv(|x| -x.powi(2) / A::cast(2.));
        (
            x * &exp,
            (x.mapv(|x| A::cast(1.) - x.powi(2)) * &exp)
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
        let alpha = A::cast(alpha);

        let gx = x.mapv(|x| (x * alpha).tanh());
        let g_x = gx.mapv(|x| alpha * (A::cast(1.) - x.powi(2)));

        Ok((gx, g_x.mean_axis(Axis(1)).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::traits::{Fit, Predict};

    use crate::hyperparams::{FastIcaParams, FastIcaValidParams};
    use ndarray_rand::rand_distr::StudentT;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<FastIca<f64>>();
        has_autotraits::<GFunc>();
        has_autotraits::<FastIcaParams<f64>>();
        has_autotraits::<FastIcaValidParams<f64>>();
        has_autotraits::<FastIcaError>();
    }

    // Test to make sure the number of components set cannot be greater
    // that the minimum of the number of rows and columns of the input
    #[test]
    fn test_ncomponents_err() {
        let input = DatasetBase::from(Array::random((4, 4), Uniform::new(0.0, 1.0)));
        let ica = FastIca::params().ncomponents(100);
        let ica = ica.fit(&input);
        assert!(ica.is_err());
    }

    // Test to make sure the alpha value of the `GFunc::Logcosh` is between
    // 1 and 2 inclusive
    #[test]
    fn test_logcosh_alpha_err() {
        let input = DatasetBase::from(Array::random((4, 4), Uniform::new(0.0, 1.0)));
        let ica = FastIca::params().gfunc(GFunc::Logcosh(10.));
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
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let source2 = Array::random_using((nsamples, 1), StudentT::new(1.0).unwrap(), &mut rng);

        // Column concatenating both the sources
        let mut sources = concatenate![Axis(1), source1.insert_axis(Axis(1)), source2];
        center_and_norm(&mut sources);

        // Mixing the two sources
        let phi: f64 = 0.6;
        let mixing = array![[phi.cos(), phi.sin()], [phi.sin(), -phi.cos()]];
        sources = mixing.dot(&sources.t());
        center_and_norm(&mut sources);

        sources = sources.reversed_axes();

        // We fit and transform using the model to unmix the two sources
        let ica = FastIca::params()
            .ncomponents(2)
            .gfunc(gfunc)
            .random_state(42);

        let sources_dataset = DatasetBase::from(sources.view());
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
