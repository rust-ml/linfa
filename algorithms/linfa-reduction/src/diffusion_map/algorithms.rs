//! Diffusion Map
//!
//! The diffusion map computes an embedding of the data by applying PCA on the diffusion operator
//! of the data. It transforms the data along the direction of the largest diffusion flow and is therefore
//! a non-linear dimensionality reduction technique. A normalized kernel describes the high dimensional
//! diffusion graph with the (i, j) entry the probability that a diffusion happens from point i to
//! j.
//!
#[cfg(not(feature = "blas"))]
use linfa_linalg::{
    eigh::*,
    lobpcg::{self, LobpcgResult, Order as TruncatedOrder},
};
use ndarray::{Array1, Array2};
#[cfg(feature = "blas")]
use ndarray_linalg::{eigh::EighInto, lobpcg, lobpcg::LobpcgResult, Scalar, TruncatedOrder, UPLO};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use linfa::dataset::{WithLapack, WithoutLapack};
use linfa::{traits::Transformer, Float};
use linfa_kernel::Kernel;
use rand::{prelude::SmallRng, SeedableRng};

use super::hyperparams::DiffusionMapValidParams;

/// Embedding of diffusion map technique
///
/// After transforming the dataset with diffusion map this structure store the embedding for
/// further use. No straightforward prediction can be made from the embedding and the algorithm
/// falls therefore in the class of transformers.
///
/// The diffusion map computes an embedding of the data by applying PCA on the diffusion operator
/// of the data. It transforms the data along the direction of the largest diffusion flow and is therefore
/// a non-linear dimensionality reduction technique. A normalized kernel describes the high dimensional
/// diffusion graph with the (i, j) entry the probability that a diffusion happens from point i to
/// j.
///
/// # Example
///
/// ```
/// use linfa::traits::Transformer;
/// use linfa_kernel::{Kernel, KernelType, KernelMethod};
/// use linfa_reduction::DiffusionMap;
///
/// let dataset = linfa_datasets::iris();
///
/// // generate sparse gaussian kernel with eps = 2 and 15 neighbors
/// let kernel = Kernel::params()
///     .kind(KernelType::Sparse(15))
///     .method(KernelMethod::Gaussian(2.0))
///     .transform(dataset.records());
///
/// // create embedding from kernel matrix using diffusion maps
/// let mapped_kernel = DiffusionMap::<f64>::params(2)
///     .steps(1)
///     .transform(&kernel)
///     .unwrap();
///
/// // get embedding from the transformed kernel matrix
/// let embedding = mapped_kernel.embedding();
/// ```
///
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionMap<F> {
    embedding: Array2<F>,
    eigvals: Array1<F>,
}

impl<'a, F: Float> Transformer<&'a Kernel<F>, DiffusionMap<F>> for DiffusionMapValidParams {
    /// Project a kernel matrix to its embedding
    ///
    /// # Parameter
    ///
    /// * `kernel`: Kernel matrix
    ///
    /// # Returns
    ///
    /// Embedding for each observation in the kernel matrix
    fn transform(&self, kernel: &'a Kernel<F>) -> DiffusionMap<F> {
        // compute spectral embedding with diffusion map
        let (embedding, eigvals) =
            compute_diffusion_map(kernel, self.steps(), 0.0, self.embedding_size(), None);

        DiffusionMap { embedding, eigvals }
    }
}

impl<F: Float> DiffusionMap<F> {
    /// Estimate the number of clusters in this embedding (very crude for now)
    pub fn estimate_clusters(&self) -> usize {
        let mean = self.eigvals.sum() / F::cast(self.eigvals.len());
        self.eigvals.iter().filter(|x| *x > &mean).count() + 1
    }

    /// Return the eigenvalue of the diffusion operator
    pub fn eigvals(&self) -> &Array1<F> {
        &self.eigvals
    }

    /// Return the embedding
    pub fn embedding(&self) -> &Array2<F> {
        &self.embedding
    }
}

#[allow(unused)]
fn compute_diffusion_map<F: Float>(
    kernel: &Kernel<F>,
    steps: usize,
    alpha: f32,
    embedding_size: usize,
    guess: Option<Array2<F>>,
) -> (Array2<F>, Array1<F>) {
    assert!(embedding_size < kernel.size());

    let d = kernel.sum().mapv(|x| x.recip());

    let (vals, vecs) = if kernel.size() < 5 * embedding_size + 1 {
        // use full eigenvalue decomposition for small problem sizes
        let mut matrix = kernel.dot(&Array2::from_diag(&d).view());
        matrix
            .columns_mut()
            .into_iter()
            .zip(d.iter())
            .for_each(|(mut a, b)| a *= *b);

        let matrix = matrix.with_lapack();
        // Calculate the eigen decomposition sorted from largest to lowest
        #[cfg(feature = "blas")]
        let (vals, vecs) = {
            let (vals, vecs) = matrix.eigh_into(UPLO::Lower).unwrap();
            (
                vals.slice_move(s![..; -1]).mapv(Scalar::from_real),
                vecs.slice_move(s![.., ..; -1]),
            )
        };
        #[cfg(not(feature = "blas"))]
        let (vals, vecs) = matrix.eigh_into().unwrap().sort_eig_desc();
        (
            vals.slice_move(s![1..=embedding_size]),
            vecs.slice_move(s![.., 1..=embedding_size]),
        )
    } else {
        let d2 = d.mapv(|x| x.powf(F::cast(0.5 + alpha)));
        // calculate truncated eigenvalue decomposition
        let x = guess
            .unwrap_or_else(|| {
                Array2::random_using(
                    (kernel.size(), embedding_size + 1),
                    Uniform::new(0.0f64, 1.0),
                    &mut SmallRng::seed_from_u64(31),
                )
                .mapv(F::cast)
            })
            .with_lapack();

        let result = lobpcg::lobpcg(
            |y| {
                let mut y = y.to_owned().without_lapack();
                y.rows_mut()
                    .into_iter()
                    .zip(d2.iter())
                    .for_each(|(mut a, b)| a *= *b);
                let mut y = kernel.dot(&y.view());

                y.rows_mut()
                    .into_iter()
                    .zip(d2.iter())
                    .for_each(|(mut a, b)| a *= *b);

                y.with_lapack()
            },
            x,
            |_| {},
            None,
            1e-7,
            200,
            TruncatedOrder::Largest,
        );

        let (vals, vecs) = match result {
            #[cfg(feature = "blas")]
            LobpcgResult::Ok(vals, vecs, _) | LobpcgResult::Err(vals, vecs, _, _) => (vals, vecs),
            #[cfg(not(feature = "blas"))]
            LobpcgResult::Ok(lobpcg) | LobpcgResult::Err((_, Some(lobpcg))) => {
                (lobpcg.eigvals, lobpcg.eigvecs)
            }
            _ => panic!("Eigendecomposition failed!"),
        };

        // cut away first eigenvalue/eigenvector
        (vals.slice_move(s![1..]), vecs.slice_move(s![.., 1..]))
    };

    let (vals, mut vecs): (Array1<F>, _) = (vals.without_lapack(), vecs.without_lapack());
    let d = d.mapv(|x| x.sqrt());

    for (mut col, val) in vecs.rows_mut().into_iter().zip(d.iter()) {
        col *= *val;
    }

    let steps = F::cast(steps);
    for (mut vec, val) in vecs.columns_mut().into_iter().zip(vals.iter()) {
        vec *= val.powf(steps);
    }

    (vecs, vals)
}
