+++
title = "Multi Class"
+++
```rust
let params = Svm::<_, Pr>::params()
    .gaussian_kernel(30.0);

// assume we have a binary decision model (here SVM) 
// predicting probability. We can merge them into a 
// multi-class model by collecting several of them
// into a `MultiClassModel`
let model = train
    .one_vs_all()?
    .into_iter()
    .map(|(l, x)| (l, params.fit(&x).unwrap()))
    .collect::<MultiClassModel<_, _>>();

// predict multi-class label
let pred = model.predict(&valid);
```
