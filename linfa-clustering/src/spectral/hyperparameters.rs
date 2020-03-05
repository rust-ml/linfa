pub enum Kernel {
    Cutoff(f32),
    Gaussian(f32)
}

pub enum Clustering {
    KMeans
}

pub struct HyperParams {
    n_clusters: usize,
    steps: usize,
    kernel: Kernel,
    clustering: Clustering,
}
