+++
title = "Gaussian Random Projection"
+++
```rust
// Assume we get some training data like MNIST: 60000 samples of 28*28 images (ie dim 784)
let dataset = Dataset::from(Array::<f64, _>::random((60000, 28 * 28), Standard));

// We can work in a reduced dimension using a Gaussian Random Projection
let reduced_dim = 100;
let proj = GaussianRandomProjection::<f32>::params()
    .target_dim(reduced_dim)
    .fit(&dataset)?;
let reduced_ds = proj.transform(&dataset);

println!("New dataset shape: {:?}", reduced_ds.records().shape());
// -> New dataset shape: [60000, 100]
```