use ndarray::{ArrayBase, Data, Array1, Ix2, Axis, DataMut};
use ndarray_rand::rand::Rng;
use ndarray_linalg::{TruncatedEig, TruncatedOrder, lobpcg::EigResult};
use super::hyperparameters::DiffusionMapHyperParams;

pub struct DiffusionMap {
    hyperparameters: DiffusionMapHyperParams,
    embedding: Array2<f64>
}

impl DiffusionMap {
    pub fn fit_predict(
        hyperparameters: DiffusionMapHyperParams,
        similarity: ArrayBase<impl DataMut<Elem = f64>, Ix2>,
        mut rng: &mut impl Rng
    ) -> Self {
        // compute spectral embedding with diffusion map
        let embedding = compute_diffusion_map(similarity, hyperparameters.steps(), hyperparameters.embedding_size());

        DiffusionMap {
            hyperparameters,
            embedding
        }
    }

    /// Return the hyperparameters used to train this spectral mode instance.
    pub fn hyperparameters(&self) -> &DiffusionMapHyperParams {
        &self.hyperparameters
    }
}

fn compute_diffusion_map(mut similarity: ArrayBase<impl DataMut<Elem = f64>, Ix2>, steps: usize, embedding_size: usize) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
    // calculate sum of rows
    let d = similarity.sum_axis(Axis(0))
        .mapv(|x| 1.0/x.sqrt());

    // ensure that matrix is symmetric
    for (idx, elm) in similarity.indexed_iter_mut() {
        let (a, b) = idx;
        *elm *= d[a] * d[b];
    }

    for val in similarity.iter_mut() {
        if val.abs() < 1e-2 {
            *val = 0.0;
        }
    }

    // calculate truncated eigenvalue decomposition
    let result = TruncatedEig::new(similarity.to_owned(), TruncatedOrder::Largest)
        .decompose(embedding_size);

    let (vals, mut vecs) = match result {
        EigResult::Ok(vals, vecs, _) | EigResult::Err(vals, vecs, _, _) => (vals, vecs),
        _ => panic!("Eigendecomposition failed!")
    };

    let d = d.mapv(|x| 1.0/x);

    for (idx, elm) in vecs.indexed_iter_mut() {
        let (row, _) = idx;
        *elm *= d[row];
    }

    for (mut vec, val) in vecs.gencolumns_mut().into_iter().zip(vals.iter()) {
        vec *= val.powf(steps as f64);
    }

    dbg!(&vals);
    vecs
}

