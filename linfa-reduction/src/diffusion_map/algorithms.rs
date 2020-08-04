use ndarray::{Array1, Array2};
use ndarray_linalg::{
    eigh::EighInto, lapack::UPLO, lobpcg, lobpcg::LobpcgResult, Scalar, TruncatedOrder,
};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use num_traits::NumCast;

use super::hyperparameters::DiffusionMapHyperParams;
use crate::kernel::{IntoKernel, Kernel};
use crate::Float;

pub struct DiffusionMap<A> {
    hyperparameters: DiffusionMapHyperParams,
    embedding: Array2<A>,
    eigvals: Array1<A>,
}

impl<A: Float> DiffusionMap<A> {
    pub fn project(hyperparameters: DiffusionMapHyperParams, kernel: impl IntoKernel<A>) -> Self {
        // compute spectral embedding with diffusion map
        let (embedding, eigvals) = compute_diffusion_map(
            kernel.into_kernel(),
            hyperparameters.steps(),
            0.0,
            hyperparameters.embedding_size(),
            None,
        );

        DiffusionMap {
            hyperparameters,
            embedding,
            eigvals,
        }
    }

    /// Return the hyperparameters used to train this spectral mode instance.
    pub fn hyperparameters(&self) -> &DiffusionMapHyperParams {
        &self.hyperparameters
    }

    /// Estimate the number of clusters in this embedding (very crude for now)
    pub fn estimate_clusters(&self) -> usize {
        let mean = self.eigvals.sum() / NumCast::from(self.eigvals.len()).unwrap();
        self.eigvals.iter().filter(|x| *x > &mean).count() + 1
    }

    /// Return a copy of the eigenvalues
    pub fn eigvals(&self) -> Array1<A> {
        self.eigvals.clone()
    }

    pub fn embedding(&self) -> Array2<A> {
        self.embedding.clone()
    }
}

fn compute_diffusion_map<A: Float>(
    kernel: impl Kernel<A>,
    steps: usize,
    alpha: f32,
    embedding_size: usize,
    guess: Option<Array2<A>>,
) -> (Array2<A>, Array1<A>) {
    assert!(embedding_size < kernel.size());

    let d = kernel.sum().mapv(|x| x.recip());

    let d2 = d.mapv(|x| x.powf(NumCast::from(0.5 + alpha).unwrap()));

    // use full eigenvalue decomposition for small problem sizes
    let (vals, mut vecs) = if kernel.size() < 5 * embedding_size + 1 {
        let mut matrix = kernel.mul_similarity(&Array2::from_diag(&d).view());
        matrix
            .gencolumns_mut()
            .into_iter()
            .zip(d.iter())
            .for_each(|(mut a, b)| a *= *b);

        let (vals, vecs) = matrix.eigh_into(UPLO::Lower).unwrap();
        let (vals, vecs) = (vals.slice_move(s![..; -1]), vecs.slice_move(s![.., ..; -1]));
        (
            vals.slice_move(s![1..embedding_size + 1])
                .mapv(|x| Scalar::from_real(x)),
            vecs.slice_move(s![.., 1..embedding_size + 1]),
        )
    } else {
        // calculate truncated eigenvalue decomposition
        let x = guess.unwrap_or(
            Array2::random(
                (kernel.size(), embedding_size + 1),
                Uniform::new(0.0f64, 1.0),
            )
            .mapv(|x| NumCast::from(x).unwrap()),
        );

        let result = lobpcg::lobpcg(
            |y| {
                let mut y = y.to_owned();
                y.genrows_mut()
                    .into_iter()
                    .zip(d2.iter())
                    .for_each(|(mut a, b)| a *= *b);
                let mut y = kernel.mul_similarity(&y.view());
                y.genrows_mut()
                    .into_iter()
                    .zip(d2.iter())
                    .for_each(|(mut a, b)| a *= *b);

                y
            },
            x,
            |_| {},
            None,
            1e-15,
            200,
            TruncatedOrder::Largest,
        );

        let (vals, vecs) = match result {
            LobpcgResult::Ok(vals, vecs, _) | LobpcgResult::Err(vals, vecs, _, _) => (vals, vecs),
            _ => panic!("Eigendecomposition failed!"),
        };

        // cut away first eigenvalue/eigenvector
        (vals.slice_move(s![1..]), vecs.slice_move(s![.., 1..]))
    };

    let d = d.mapv(|x| x.sqrt());

    for (mut col, val) in vecs.genrows_mut().into_iter().zip(d.iter()) {
        col *= *val;
    }

    let steps = NumCast::from(steps).unwrap();
    for (mut vec, val) in vecs.gencolumns_mut().into_iter().zip(vals.iter()) {
        vec *= val.powf(steps);
    }

    (vecs, vals)
}
