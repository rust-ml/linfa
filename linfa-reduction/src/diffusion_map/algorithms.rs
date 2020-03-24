use ndarray::{ArrayBase, Array1, Array2, Ix2, Axis, DataMut};
use ndarray_linalg::{TruncatedEig, TruncatedOrder, lobpcg::LobpcgResult, lobpcg, eigh::EighInto, lapack::UPLO, close_l2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::kernel::{Kernel, IntoKernel};
use super::hyperparameters::DiffusionMapHyperParams;

pub struct DiffusionMap {
    hyperparameters: DiffusionMapHyperParams,
    embedding: Array2<f64>,
    eigvals: Array1<f64>
}

impl DiffusionMap {
    pub fn project(
        hyperparameters: DiffusionMapHyperParams,
        kernel: impl IntoKernel<f64>
    ) -> Self {
        // compute spectral embedding with diffusion map
        let (embedding, eigvals) = compute_diffusion_map(kernel.into_kernel(), hyperparameters.steps(), hyperparameters.embedding_size());

        DiffusionMap {
            hyperparameters,
            embedding,
            eigvals
        }
    }

    /// Return the hyperparameters used to train this spectral mode instance.
    pub fn hyperparameters(&self) -> &DiffusionMapHyperParams {
        &self.hyperparameters
    }

    /// Estimate the number of clusters in this embedding (very crude for now)
    pub fn estimate_clusters(&self) -> usize {
        let mean = self.eigvals.sum() / self.eigvals.len() as f64;
        self.eigvals.iter().filter(|x| *x > &mean).count() + 1
    }

    /// Return a copy of the eigenvalues
    pub fn eigvals(&self) -> Array1<f64> {
        self.eigvals.clone()
    }

    /// Return the embedding
    pub fn embedding(self) -> Array2<f64> {
        self.embedding
    }
}

fn compute_diffusion_map(kernel: impl Kernel<f64>, steps: usize, embedding_size: usize) -> (Array2<f64>, Array1<f64>) {
    assert!(embedding_size < kernel.size());

    // calculate sum of rows
    let d = kernel.sum()
        .mapv(|x| 1.0/x.sqrt());

    // use full eigenvalue decomposition for small problem sizes
    let (vals, mut vecs) = if kernel.size() < 5 * embedding_size + 1 {
        let mut matrix = kernel.apply_gram(Array2::from_diag(&d));
        matrix.genrows_mut().into_iter().zip(d.iter()).for_each(|(mut c, x)| c *= *x);

        let (vals, vecs) = matrix.eigh_into(UPLO::Lower).unwrap();
        let (vals, vecs) = (vals.slice_move(s![..; -1]), vecs.slice_move(s![.., ..; -1]));
        (
            vals.slice_move(s![1..embedding_size+1]),
            vecs.slice_move(s![..,1..embedding_size+1])
        )
    } else {
        // calculate truncated eigenvalue decomposition
        let x = Array2::random((d.len(), embedding_size + 1), Uniform::new(0.0, 1.0));

        let result = lobpcg::lobpcg(|y| {
            let mut y = y.to_owned();
            y.genrows_mut().into_iter().zip(d.iter()).for_each(|(mut c, x)| c *= *x);
            let mut y = kernel.apply_gram(y);
            y.genrows_mut().into_iter().zip(d.iter()).for_each(|(mut c, x)| c *= *x);

            y
        }, x, |_| {}, None, 1e-5, 20, TruncatedOrder::Largest);

        let (vals, vecs) = match result {
            LobpcgResult::Ok(vals, vecs, _) | LobpcgResult::Err(vals, vecs, _, _) => (vals, vecs),
            _ => panic!("Eigendecomposition failed!")
        };

        // cut away first eigenvalue/eigenvector
        (
            vals.slice_move(s![1..]),
            vecs.slice_move(s![..,1..])
        )
    };
    let d = d.mapv(|x| 1.0/x);

    for (idx, elm) in vecs.indexed_iter_mut() {
        let (row, _) = idx;
        *elm *= d[row];
    }

    for (mut vec, val) in vecs.gencolumns_mut().into_iter().zip(vals.iter()) {
        vec *= val.powf(steps as f64);
    }

    (vecs, vals)
}

