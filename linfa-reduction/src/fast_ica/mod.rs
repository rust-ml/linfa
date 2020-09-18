use ndarray::{Array, Array1, Array2, Axis};
use ndarray_linalg::{eigh::Eigh, lapack::UPLO, svd::SVD};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;

use crate::Float;

pub enum Algorithm {
    Parallel,
    Deflation,
}

pub enum GFunc {
    Logcosh(f64),
    Exp,
    Cube,
}

pub struct FastIca {
    n_components: usize,
    algorithm: Algorithm,
    gfunc: GFunc,
    max_iter: usize,
    tol: f64,
}

impl FastIca {
    pub fn new(n_components: usize) -> Self {
        FastIca {
            n_components,
            algorithm: Algorithm::Parallel,
            gfunc: GFunc::Logcosh(1.),
            max_iter: 200,
            tol: 1e-4,
        }
    }

    pub fn set_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
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

    pub fn fit<A: Float>(&self, x: &Array2<A>) -> FittedFastIca<A> {
        let shape = x.shape();
        let (n_samples, n_features) = (shape[0], shape[1]);

        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let x_mean = x_mean.insert_axis(Axis(1));

        let x = x.t().to_owned();
        let x = x - &x_mean;

        // TODO: Validate `n_components`
        // TODO: Validate `GFunc::Logcosh`'s alpha value

        // TODO: `k` creation should be more legible
        let (u, s, _) = x.svd(true, false).unwrap();
        let u = u.unwrap();
        let u = u.slice(s![.., ..n_samples.min(n_features)]).to_owned();
        let s = s.mapv(|x| A::from(x).unwrap());
        let k = u / s;
        let k = k.t();
        let k = k.slice(s![..self.n_components, ..]);
        let x1 = k.dot(&x);
        let nfeatures_sqrt = (n_features as f64).sqrt();
        x1.mapv(|x| x * A::from(nfeatures_sqrt).unwrap());

        // TODO: Seed the random generated array
        let w_init = Array::random((self.n_components, self.n_components), Uniform::new(0., 1.));
        let w_init = w_init.mapv(|x| A::from(x).unwrap());

        let w = match self.algorithm {
            Algorithm::Parallel => self.ica_parallel(&x1, &w_init),
            Algorithm::Deflation => todo!(),
        };

        let components = w.dot(&k);

        FittedFastIca {
            mean: x_mean.t().to_owned(),
            components,
        }
    }

    fn ica_parallel<A: Float>(&self, x: &Array2<A>, w_init: &Array2<A>) -> Array2<A> {
        let mut w = Self::sym_decorrelation(&w_init);
        let p = x.shape()[1] as f64;

        for _ in 0..self.max_iter {
            let (gwtx, g_wtx) = match self.gfunc {
                GFunc::Cube => Self::cube(&w.dot(x)),
                GFunc::Exp => todo!(),
                GFunc::Logcosh(_alpha) => todo!(),
            };

            let lhs = gwtx.dot(&x.t()).mapv(|x| x / A::from(p).unwrap());
            let rhs = &w * &g_wtx.insert_axis(Axis(1));
            let w1 = Self::sym_decorrelation(&(lhs - rhs));

            // TODO: Find a better way
            let lim = w1.dot(&w.t());
            let lim = lim.diag();
            let lim = lim.mapv(num_traits::Float::abs);
            let lim = lim.mapv(|x| x - A::from(1.).unwrap());
            let lim = lim.mapv(num_traits::Float::abs);
            let lim = lim.max().unwrap();

            w = w1;

            if lim < &A::from(self.tol).unwrap() {
                break;
            }
        }

        w
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

    fn cube<A: Float>(x: &Array2<A>) -> (Array2<A>, Array1<A>) {
        (
            x.mapv(|x| x.powi(3)),
            x.mapv(|x| A::from(3.).unwrap() * x.powi(2))
                .mean_axis(Axis(1))
                .unwrap(),
        )
    }
}

pub struct FittedFastIca<A> {
    mean: Array2<A>,
    components: Array2<A>,
}

impl<A: Float> FittedFastIca<A> {
    pub fn transform(&self, x: &Array2<A>) -> Array2<A> {
        let x = x - &self.mean;
        x.dot(&self.components.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample() {
        let a = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];

        let ica = FastIca::new(2).set_gfunc(GFunc::Cube);
        let ica = ica.fit(&a);
        let x = ica.transform(&a);
        println!("{:?}", x);

        assert!(false);
    }
}
