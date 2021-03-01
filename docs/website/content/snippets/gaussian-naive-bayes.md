+++
title = "Gaussian Naive Bayes"
+++
```rust
let (train, valid) = linfa_datasets::iris()
    .split_with_ratio(0.8);

// train the model
let model = GaussianNbParams::params()
    .fit(&train)?;

// Predict the validation dataset
let pred = model.predict(&valid);

// construct confusion matrix
let cm = pred.confusion_matrix(&valid)?;

// print confusion matrix, accuracy and precision
println!("{:?}", cm);
println!("accuracy {}, precision {}", 
    cm.accuracy(), cm.precision());
```
