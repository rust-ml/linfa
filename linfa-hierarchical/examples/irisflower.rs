use std::error::Error;

use linfa::traits::Transformer;
use linfa_hierarchical::HierarchicalCluster;
use linfa_kernel::{Kernel, KernelMethod};

fn main() -> Result<(), Box<dyn Error>> {
    // load Iris plant dataset
    let dataset = linfa_datasets::iris();

    let kernel = Kernel::params()
        .method(KernelMethod::Gaussian(1.0))
        .transform(dataset.records().view());

    let kernel = HierarchicalCluster::default()
        .num_clusters(3)
        .transform(kernel);

    for (id, target) in kernel
        .targets()
        .into_iter()
        .zip(dataset.targets().into_iter())
    {
        let name = match *target as usize {
            0 => "setosa",
            1 => "versicolor",
            2 => "virginica",
            _ => unreachable!(),
        };

        print!("({} {}) ", id, name);
    }
    println!("");

    Ok(())
}
