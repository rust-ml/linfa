use ndarray::{ArrayBase, Data, Array1, Ix2, Axis, DataMut};
use ndarray_rand::rand::Rng;
use ndarray_linalg::{eigh::Eigh, lapack::UPLO};

use crate::k_means::*;
use super::hyperparameters::SpectralClusteringHyperParams;

pub struct SpectralClustering {
    hyperparameters: SpectralClusteringHyperParams,
    indices: Array1<usize>
}

impl SpectralClustering {
    pub fn fit_predict(
        hyperparameters: SpectralClusteringHyperParams,
        similarity: ArrayBase<impl DataMut<Elem = f64>, Ix2>,
        mut rng: &mut impl Rng
    ) -> Self {
        // compute spectral embedding with diffusion map
        let embedding = compute_diffusion_map(similarity, hyperparameters.steps(), hyperparameters.embedding_size());
        dbg!(&embedding);

        // calculate centroids of this embedding
        let conf = KMeansHyperParams::new(hyperparameters.n_clusters())
            .build();

        let kmeans = KMeans::fit(conf, &embedding, &mut rng);
        let indices = kmeans.predict(&embedding);

        SpectralClustering {
            hyperparameters,
            indices
        }
    }

    /// Return the hyperparameters used to train this spectral mode instance.
    pub fn hyperparameters(&self) -> &SpectralClusteringHyperParams {
        &self.hyperparameters
    }

    /// Return the indices
    pub fn indices(&self) -> Array1<usize> {
        self.indices.clone()
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
    //dbg!(&similarity);*/

    // calculate eigenvalue decomposition
    let (vals, mut vecs) = similarity.eigh(UPLO::Upper).unwrap();

    let n_irrelevant = vals.iter().filter(|x| (*x-1.0).abs() < 1e-2).count();
    let embedding_size = usize::min(similarity.len_of(Axis(0)) - n_irrelevant, embedding_size);
    let (start, end) = (similarity.len_of(Axis(0)) - embedding_size - n_irrelevant, similarity.len_of(Axis(0)) - n_irrelevant);

    let d = d.mapv(|x| 1.0/x);

    for (idx, elm) in vecs.indexed_iter_mut() {
        let (row, _) = idx;
        *elm *= d[row];
    }

    // crop eigenvectors to wished embedding dimension

    let vals = vals.slice(s![start..end]);
    let mut vecs = vecs.slice(s![start..end, ..]).t().into_owned();

    for (mut vec, val) in vecs.gencolumns_mut().into_iter().zip(vals.iter()) {
        vec *= val.powf(steps as f64);
    }

    vecs
}

