use linfa::{error::Result, traits::Transformer};
use linfa_kernel::{Kernel, KernelMethod, KernelType};
use linfa_reduction::utils::generate_convoluted_rings2d;
use linfa_reduction::DiffusionMap;

use ndarray_npy::write_npy;
use rand::{rngs::SmallRng, SeedableRng};

fn main() -> Result<()> {
    // Our random number generator, seeded for reproducibility
    let mut rng = SmallRng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let n = 500;

    // generate three convoluted rings
    let dataset =
        generate_convoluted_rings2d(&[(0.0, 3.0), (10.0, 13.0), (20.0, 23.0)], n, &mut rng);

    // generate sparse polynomial kernel with k = 14, c = 5 and d = 2
    let kernel = Kernel::params()
        //.method(KernelMethod::Polynomial(5.0, 2.0))
        .kind(KernelType::Sparse(15))
        .method(KernelMethod::Gaussian(2.0))
        //.kind(KernelType::Dense)
        .transform(dataset.view());

    let embedding = DiffusionMap::<f64>::params(2).steps(1).transform(&kernel)?;

    // get embedding
    let embedding = embedding.embedding();

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("diffusion_map_dataset.npy", dataset).expect("Failed to write .npy file");
    write_npy("diffusion_map_embedding.npy", embedding.to_owned())
        .expect("Failed to write .npy file");

    Ok(())
}
