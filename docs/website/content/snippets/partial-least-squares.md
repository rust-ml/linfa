+++
title = "Partial least squares regression"
+++
```rust
// Load linnerud dataset with  20 samples, 
// 3 input features, 3 output features
let ds = linfa_datasets::linnerud();

// Fit PLS2 method using 2 principal components 
// (latent variables)
let pls = PlsRegression::params(2).fit(&ds)?;

// We can either apply the dimension reduction to the dataset
let reduced_ds = pls.transform(ds);

// ... or predict outputs given a new input sample.
let exercices = array![[14., 146., 61.], [6., 80., 60.]];
let physio_measures = pls.predict(exercices);
```
