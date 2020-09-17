use ndarray::{Array, Array1, Array2, Axis};
use ndarray_linalg::{eigh::Eigh, lapack::UPLO, svd::SVD};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;

use crate::Float;

enum Algorithm {
    Parallel,
    Deflation,
}

enum GFunc {
    Logcosh(f64),
    Exp,
    Cube,
}

struct FastIca {
    n_components: usize,
    algorithm: Algorithm,
    gfunc: GFunc,
    max_iter: usize,
    tol: f64,
}

impl FastIca {
    fn new(n_components: usize) -> Self {
        FastIca {
            n_components,
            algorithm: Algorithm::Parallel,
            gfunc: GFunc::Logcosh(1.),
            max_iter: 200,
            tol: 1e-4,
        }
    }

    fn set_algorithm(&mut self, algorithm: Algorithm) {
        self.algorithm = algorithm;
    }

    fn set_gfunc(&mut self, gfunc: GFunc) {
        self.gfunc = gfunc;
    }

    fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    fn set_tol(&mut self, tol: f64) {
        self.tol = tol;
    }

    fn fit<A: Float>(&self, x: Array2<A>) {
        let shape = x.shape();
        let (n_samples, n_features) = (shape[0], shape[1]);

        // TODO: Validate `n_components`
        // TODO: Validate `GFunc::Logcosh`'s alpha value

        let x_mean = x.mean_axis(Axis(1)).unwrap();
        let x = x - x_mean.insert_axis(Axis(1));

        // TODO: `k` creation should be more legible
        let (u, s, _) = x.svd(true, false).unwrap();
        let u = u.unwrap();
        let s = s.mapv(|x| A::from(x).unwrap());
        let k = u / s;
        let k = k.t();
        let k = k.slice(s![..2, ..]);

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
            let rhs = g_wtx.insert_axis(Axis(1)) * &w;
            let w1 = Self::sym_decorrelation(&(lhs - rhs));

            // TODO: Find a better way
            let lim = w1.dot(&w.t());
            let lim = lim.diag();
            let lim = lim.mapv(num_traits::Float::abs);
            let lim = lim.mapv(|x| x - A::from(1.).unwrap());
            let lim = lim.mapv(num_traits::Float::abs);
            let lim = lim.max().unwrap();

            w = w1;

            if lim > &A::from(self.tol).unwrap() {
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

