+++
title = "Release 0.3.1"
date = "2021-03-11"
+++

In this release of Linfa the documentation is extended, new examples are added and the functionality of datasets improved. No new algorithms were added.

<!-- more -->

The meta-issue [#82](https://github.com/rust-ml/linfa/issues/82) gives a good overview of the necessary documentation improvements and testing/documentation/examples were considerably extended in this release. 

Further new functionality was added to datasets and multi-target datasets are introduced. Bootstrapping is now possible for features and samples and you can cross-validate your model with k-folding. We polished various bits in the kernel machines and simplified the interface there.

The trait structure of regression metrics are simplified and the silhouette score introduced for easier testing of K-Means and other algorithms.


# Changes

 * improve documentation in all algorithms, various commits
 * add a website to the infrastructure (c8acc785b)
 * add k-folding with and without copying (b0af80546f8)
 * add feature naming and pearson's cross correlation (71989627f)
 * improve ergonomics when handling kernels (1a7982b973)
 * improve TikZ generator in `linfa-trees` (9d71f603bbe)
 * introduce multi-target datasets (b231118629)
 * simplify regression metrics and add cluster metrics (d0363a1fa8ef)

# Example

You can now perform cross-validation with k-folding. @Sauro98 actually implemented two versions, one which copies the dataset into k folds and one which avoid excessive memory operations by copying only the validation dataset around. For example to test a model with 8-folding:

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
