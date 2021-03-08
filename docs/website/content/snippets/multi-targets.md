+++
title = "Multi Targets"
+++
```rust
// assume we have a dataset with multiple,
// uncorrelated targets and we want to train
// a single model for each target variable
let model = train.target_iter()
    .map(|x| params.fit(&x).unwrap())
    .collect::<MultiTarget<_, _>>()?;

// composing `model` returns multiple targets
let valid_est = model.predict(valid);
println!("{}", valid_est.ntargets());
```
