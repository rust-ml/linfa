+++
title = "Cross Validation"
+++
```rust
// perform cross-validation with the MCC
let mcc_runs = linfa_datasets::diabetes()
    .iter_fold(6, |v| params.fit(&v).unwrap())
    .map(|(model, valid)| {
        let y_est = model.predict(&valid)
        valid.confusion_matrix(y_est)
    })
    .map(|x| x.mcc())
    .collect::<Array1<_>>();

// calculate mean and standard deviation
println!("MCC: {}±{}",
    mcc_runs.mean(),
    mcc_runs.stddev()
);
```
