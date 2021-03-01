+++
title = "DecisionTrees"
+++
```rust
let (train, valid) = linfa_datasets::iris()
    .split_with_ratio(0.8);

// Train model with Gini criterion
let gini_model = DecisionTree::params()
    .split_quality(SplitQuality::Gini)
    .max_depth(Some(100))
    .min_weight_split(1.0)
    .fit(&train)?;

let cm = gini_model.predict(&valid)
    .confusion_matrix(&valid);

println!("{:?}", cm);
println!("Accuracy {}%", cm.accuracy() * 100.0);
```
