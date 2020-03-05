pub struct SpectralClustering {
    hyperparameters: HyperParams,
}

impl SpectralClustering {
    pub fn fit(
        hyperparameters: HyperParams,
        observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
        rng: &mut impl Rng
    ) -> Self {
        // compute spectral embedding with diffusion map
        let embedding = compute_diffusion_map(observations);
        // calculate centroids of this embedding
        let kmeans = KMeans::fit(&embedding, &mut rng);

        SpectralClustering {
            hyperparameters,
            observations,
            embedding,
            kmeans
        }
    }

    pub fn predict(&self, observations: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array1<usize> {
        // choose nearest observations and its embeddings
        let indices = choose_nearest(&self.observations, observations);
        let embeddings = self.embeddings[indices];

        self.kmeans.predict(embeddings)
    }

    /// Return the hyperparameters used to train this spectral mode instance.
    pub fn hyperparameters(&self) -> &HyperParams {
        &self.hyperparams
    }
}
