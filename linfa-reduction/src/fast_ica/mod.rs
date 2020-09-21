use ndarray::{Array, Array1, Array2, Axis};
use ndarray_linalg::{eigh::Eigh, lapack::UPLO, svd::SVD};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;
use rand_isaac::Isaac64Rng;

use crate::Float;

#[derive(Debug)]
pub struct FastIca {
    ncomponents: Option<usize>,
    gfunc: GFunc,
    max_iter: usize,
    tol: f64,
    random_state: Option<usize>,
}

impl Default for FastIca {
    fn default() -> Self {
        Self::new()
    }
}

impl FastIca {
    pub fn new() -> Self {
        FastIca {
            ncomponents: None,
            gfunc: GFunc::Logcosh(1.),
            max_iter: 200,
            tol: 1e-4,
            random_state: None,
        }
    }

    pub fn set_ncomponents(mut self, ncomponents: usize) -> Self {
        self.ncomponents = Some(ncomponents);
        self
    }

    pub fn set_gfunc(mut self, gfunc: GFunc) -> Self {
        self.gfunc = gfunc;
        self
    }

    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn set_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn set_random_state(mut self, random_state: usize) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl FastIca {
    pub fn fit<A: Float>(&self, x: &Array2<A>) -> Result<FittedFastIca<A>, String> {
        let (nsamples, nfeatures) = (x.nrows(), x.ncols());

        let ncomponents = self.ncomponents.unwrap_or_else(|| nsamples.min(nfeatures));
        if ncomponents > nsamples.min(nfeatures) {
            return Err(
                "ncomponents cannot be greater than the min(sample size, features size)"
                    .to_string(),
            );
        }

        let x_mean = x.mean_axis(Axis(0)).unwrap();

        let mut x_centered = x - &x_mean.to_owned().insert_axis(Axis(0));
        x_centered = x_centered.reversed_axes();

        // TODO: Propogating convergence error
        // TODO: Error handling

        let k = match x_centered.svd(true, false).unwrap() {
            (Some(u), s, _) => {
                let s = s.mapv(|x| A::from(x).unwrap());
                (u.slice(s![.., ..nsamples.min(nfeatures)]).to_owned() / s)
                    .t()
                    .slice(s![..ncomponents, ..])
                    .to_owned()
            }
            _ => unreachable!(),
        };
        let mut x_whitened = k.dot(&x_centered);
        let nsamples_sqrt = A::from((nsamples as f64).sqrt()).unwrap();
        x_whitened.mapv_inplace(|x| x * nsamples_sqrt);

        // TODO: Seed the random generated array
        let w_init: Array2<f64>;
        if let Some(seed) = self.random_state {
            let mut rng = Isaac64Rng::seed_from_u64(seed as u64);
            w_init =
                Array::random_using((ncomponents, ncomponents), Uniform::new(0., 1.), &mut rng);
        } else {
            w_init = Array::random((ncomponents, ncomponents), Uniform::new(0., 1.));
        }
        let w_init = w_init.mapv(|x| A::from(x).unwrap());

        let w = self.ica_parallel(&x_whitened, &w_init)?;

        let components = w.dot(&k);

        Ok(FittedFastIca {
            mean: x_mean,
            components,
        })
    }

    fn ica_parallel<A: Float>(
        &self,
        x: &Array2<A>,
        w_init: &Array2<A>,
    ) -> Result<Array2<A>, String> {
        let mut w = Self::sym_decorrelation(&w_init);
        let p = x.shape()[1] as f64;

        for _ in 0..self.max_iter {
            let (gwtx, g_wtx) = self.gfunc.exec(&w.dot(x))?;

            let lhs = gwtx.dot(&x.t()).mapv(|x| x / A::from(p).unwrap());
            let rhs = &w * &g_wtx.insert_axis(Axis(1));
            let w_new = Self::sym_decorrelation(&(lhs - rhs));

            let lim = *w_new
                .dot(&w.t())
                .diag()
                .mapv(num_traits::Float::abs)
                .mapv(|x| x - A::from(1.).unwrap())
                .mapv(num_traits::Float::abs)
                .max()
                .unwrap();

            w = w_new;

            if lim < A::from(self.tol).unwrap() {
                break;
            }
        }

        Ok(w)
    }

    fn sym_decorrelation<A: Float>(w: &Array2<A>) -> Array2<A> {
        let (eig_val, eig_vec) = w.dot(&w.t()).eigh(UPLO::Upper).unwrap();
        let eig_val = eig_val.mapv(|x| A::from(x).unwrap());

        let tmp = &eig_vec
            * &(eig_val
                .mapv(num_traits::Float::sqrt)
                .mapv(num_traits::Float::recip))
            .insert_axis(Axis(0));

        tmp.dot(&eig_vec.t()).dot(w)
    }
}

#[derive(Debug)]
pub struct FittedFastIca<A> {
    mean: Array1<A>,
    components: Array2<A>,
}

impl<A: Float> FittedFastIca<A> {
    pub fn transform(&self, x: &Array2<A>) -> Array2<A> {
        let x_centered = x - &self.mean.to_owned().insert_axis(Axis(0));
        x_centered.dot(&self.components.t())
    }
}

#[derive(Debug)]
pub enum GFunc {
    Logcosh(f64),
    Exp,
    Cube,
}

impl GFunc {
    fn exec<A: Float>(&self, x: &Array2<A>) -> Result<(Array2<A>, Array1<A>), String> {
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

    fn logcosh<A: Float>(x: &Array2<A>, alpha: f64) -> Result<(Array2<A>, Array1<A>), String> {
        if !(alpha >= 1. && alpha <= 2.) {
            return Err("alpha must be between 1 and 2 inclusive".to_string());
        }
        let alpha = A::from(alpha).unwrap();
        let gx = x.mapv(|x| (x * alpha).tanh());

        let g_x = gx.mapv(|x| alpha * (A::from(1.).unwrap() - x.powi(2)));

        Ok((gx, g_x.mean_axis(Axis(1)).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray_rand::rand_distr::StudentT;

    #[test]
    fn test_ncomponents_err() {
        let input = Array::random((4, 4), Uniform::new(0.0, 1.0));
        let ica = FastIca::new().set_ncomponents(100);
        let ica = ica.fit(&input);
        assert!(ica.is_err());
    }

    #[test]
    fn test_logcosh_alpha_err() {
        let input = Array::random((4, 4), Uniform::new(0.0, 1.0));
        let ica = FastIca::new().set_gfunc(GFunc::Logcosh(10.));
        let ica = ica.fit(&input);
        assert!(ica.is_err());
    }

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

    fast_ica_tests! {
        exp: GFunc::Exp, cube: GFunc::Cube, logcosh: GFunc::Logcosh(1.0),
    }

    fn test_fast_ica(gfunc: GFunc) {
        let n_samples = 1000;

        // mean and norm
        let center_and_norm = |s: &mut Array2<f64>| {
            let mean = s.mean_axis(Axis(0)).unwrap();
            *s -= &mean.insert_axis(Axis(0));
            let std = s.std_axis(Axis(0), 0.);
            *s /= &std.insert_axis(Axis(0));
        };

        // sin with linspace for n_samples
        let mut s1 = Array::linspace(0., 100., n_samples);
        s1.mapv_inplace(|x| {
            let tmp = 2. * f64::sin(x);
            if tmp > 0. {
                return 0.;
            }
            -1.
        });

        // students t random matrix for n_samples
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let s2 = Array::random_using((n_samples, 1), StudentT::new(1.0).unwrap(), &mut rng);

        // column stacking
        let mut s = stack![Axis(1), s1.insert_axis(Axis(1)), s2];
        center_and_norm(&mut s);

        let phi: f64 = 0.6;
        let mixing = array![[phi.cos(), phi.sin()], [phi.sin(), -phi.cos()]];
        s = mixing.dot(&s.t());
        center_and_norm(&mut s);
        s = s.reversed_axes();

        let ica = FastIca::new()
            .set_ncomponents(2)
            .set_gfunc(gfunc)
            .set_random_state(42);
        let ica = ica.fit(&s).unwrap();
        let mut s_ = ica.transform(&s);
        center_and_norm(&mut s_);
        assert_eq!(s_.shape(), &[1000, 2]);

        // Accounting for the ambiguity in the order
        // and the sign
        let s1 = s.column(0);
        let s2 = s.column(1);
        let mut s1_ = s_.column(0);
        let mut s2_ = s_.column(1);
        if s1_.dot(&s2).abs() > s1_.dot(&s1).abs() {
            s1_ = s_.column(1);
            s2_ = s_.column(0);
        }

        let similarity1 = s1.dot(&s1_).abs() / (s.nrows() as f64);
        let similarity2 = s2.dot(&s2_).abs() / (s.nrows() as f64);
        assert!(similarity1.max(similarity2) > 0.9);
    }
}
