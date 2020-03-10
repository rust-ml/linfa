pub enum Clustering {
    KMeans
}
pub struct SpectralClusteringHyperParams {
    n_clusters: usize,
    steps: usize,
    embedding_size: usize,
    clustering: Clustering,
}

pub struct SpectralClusteringHyperParamsBuilder {
    n_clusters: usize,
    steps: usize,
    embedding_size: usize,
    clustering: Clustering,
}

impl SpectralClusteringHyperParamsBuilder {
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;

        self
    }

    pub fn clustering(mut self, clustering: Clustering) -> Self {
        self.clustering = clustering;

        self
    }

    pub fn build(self) -> SpectralClusteringHyperParams {
        SpectralClusteringHyperParams::build(
            self.n_clusters,
            self.steps,
            self.embedding_size,
            self.clustering
        )
    }
}

impl SpectralClusteringHyperParams {
    pub fn new(n_clusters: usize, embedding_size: usize) -> SpectralClusteringHyperParamsBuilder {
        SpectralClusteringHyperParamsBuilder {
            steps: 10,
            clustering: Clustering::KMeans,
            n_clusters,
            embedding_size
        }
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    pub fn build(n_clusters: usize, steps: usize, embedding_size: usize, clustering: Clustering) -> Self {
        SpectralClusteringHyperParams {
            steps,
            n_clusters,
            embedding_size,
            clustering
        }
    }
}
