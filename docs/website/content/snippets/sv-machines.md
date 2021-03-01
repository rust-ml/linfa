+++
title = "Support Vector Machines"
+++
```rust
// everything above 6.5 is considered a good wine
let (train, valid) = linfa_datasets::winequality()
    .map_targets(|x| *x > 6)
    .split_with_ratio(0.9);

// train SVM with nu=0.01 and RBF with eps=80.0
let model = Svm::params()
    .nu_weight(0.01)
    .gaussian_kernel(80.0)
    .fit(&train)?;

// print model performance and number of SVs
println!("{}", model);
```
