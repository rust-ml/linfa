+++
title = "K folding"
+++
```rust
// perform cross-validation with the F1 score
let f1_runs = dataset
    .iter_fold(8, |v| params.fit(&v).unwrap())
    .map(|(model, valid)| {
        let cm = model
            .predict(&valid)
            .mapv(|x| x > Pr::even())
            .confusion_matrix(&valid).unwrap();
  
          cm.f1_score()
    })  
    .collect::<Array1<_>>();
  
// calculate mean and standard deviation
println!("F1 score: {}±{}",
    f1_runs.mean().unwrap(),
    f1_runs.std_axis(Axis(0), 0.0),
);  
```
