+++
title = "Diffusion Maps"
+++
```rust
// generate RBF kernel with sparsity constraints
let kernel = Kernel::params()
    .kind(KernelType::Sparse(15))
    .method(KernelMethod::Gaussian(2.0))
    .transform(dataset.view());

let embedding = DiffusionMap::<f64>::params(2)
    .steps(1)
    .transform(&kernel)?;

// get embedding
let embedding = embedding.embedding();
```
