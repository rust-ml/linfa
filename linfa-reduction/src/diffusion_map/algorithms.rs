use ndarray::{Array1, Array2, ArrayView2};
use ndarray_linalg::{
    eigh::EighInto, lapack::UPLO, lobpcg, lobpcg::LobpcgResult, Scalar, TruncatedOrder, Lapack
};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use num_traits::NumCast;

use linfa::{Float, traits::Transformer};
use linfa_kernel::Kernel;

use super::hyperparameters::{DiffusionMapHyperParams, DiffusionMapHyperParamsBuilder};

pub struct DiffusionMap<F> {
    embedding: Array2<F>,
    eigvals: Array1<F>,
}

impl<'a, F: Float + Lapack> Transformer<&'a Kernel<ArrayView2<'a, F>>, DiffusionMap<F>> for DiffusionMapHyperParams {
    fn transform(
        &self,
        kernel: &'a Kernel<ArrayView2<'a, F>>,
    ) -> DiffusionMap<F> {
        // compute spectral embedding with diffusion map
        let (embedding, eigvals) = compute_diffusion_map(
            kernel,
            self.steps(),
            0.0,
            self.embedding_size(),
            None,
        );

        DiffusionMap {
            embedding,
            eigvals,
        }
    }
}

impl<F: Float + Lapack> DiffusionMap<F> {
    pub fn params(embedding_size: usize) -> DiffusionMapHyperParamsBuilder {
        DiffusionMapHyperParams::new(embedding_size)
    }
    /// Estimate the number of clusters in this embedding (very crude for now)
    pub fn estimate_clusters(&self) -> usize {
        let mean = self.eigvals.sum() / NumCast::from(self.eigvals.len()).unwrap();
        self.eigvals.iter().filter(|x| *x > &mean).count() + 1
    }

    /// Return a copy of the eigenvalues
    pub fn eigvals(&self) -> Array1<F> {
        self.eigvals.clone()
    }

    pub fn embedding(&self) -> Array2<F> {
        self.embedding.clone()
    }
}

fn compute_diffusion_map<'b, F: Float + Lapack>(
    kernel: &'b Kernel<ArrayView2<'b, F>>,
    steps: usize,
    alpha: f32,
    embedding_size: usize,
    guess: Option<Array2<F>>,
) -> (Array2<F>, Array1<F>) {
    assert!(embedding_size < kernel.size());

    let d = kernel.sum().mapv(|x| x.recip());

    let d2 = d.mapv(|x| x.powf(NumCast::from(0.5 + alpha).unwrap()));

    // use full eigenvalue decomposition for small problem sizes
    let (vals, mut vecs) = if kernel.size() < 5 * embedding_size + 1 {
        let mut matrix = kernel.dot(&Array2::from_diag(&d).view());
        matrix
            .gencolumns_mut()
            .into_iter()
            .zip(d.iter())
            .for_each(|(mut a, b)| a *= *b);

        let (vals, vecs) = matrix.eigh_into(UPLO::Lower).unwrap();
        let (vals, vecs) = (vals.slice_move(s![..; -1]), vecs.slice_move(s![.., ..; -1]));
        (
            vals.slice_move(s![1..=embedding_size])
                .mapv(Scalar::from_real),
            vecs.slice_move(s![.., 1..=embedding_size]),
        )
    } else {
        // calculate truncated eigenvalue decomposition
        let x = guess.unwrap_or_else(|| {
            Array2::random(
                (kernel.size(), embedding_size + 1),
                Uniform::new(0.0f64, 1.0),
            )
            .mapv(|x| NumCast::from(x).unwrap())
        });

        let result = lobpcg::lobpcg(
            |y| {
                let mut y = y.to_owned();
                y.genrows_mut()
                    .into_iter()
                    .zip(d2.iter())
                    .for_each(|(mut a, b)| a *= *b);
                let mut y = kernel.dot(&y.view());
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
