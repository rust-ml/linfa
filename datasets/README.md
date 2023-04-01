# Datasets

`linfa-datasets` provides a collection of commonly used datasets ready to be used in tests and examples.

## The Big Picture

`linfa-datasets` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current State

Currently the following datasets are provided:

| Name | Description | #samples, #features, #targets | Targets | Reference |
| :--- | :--- | :---| :--- | :--- |
| iris | The Iris dataset provides samples of flower properties, belonging to three different classes. Only two of them are linearly separable. It was introduced by Ronald Fisher in 1936 as an example for linear discriminant analysis. |  150, 4, 1 | Multi-class classification | [here](https://archive.ics.uci.edu/ml/datasets/iris) |
| winequality | The winequality dataset measures different properties of wine, such as acidity, and gives a scoring from 3 to 8 in quality. It was collected in the north of Portugal. | 441, 10, 1 | Multi-class classification | [here](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| diabetes | The diabetes dataset gives samples of human biological measures, such as BMI, age, blood measures, and tries to predict the progression of diabetes. | 1599, 11, 1 | Regression | [here](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) |
| linnerud | The linnerud dataset contains samples from 20 middle-aged men in a fitness club. Their physical capability, as well as biological measures are related. | 20, 3, 3 | Regression | [here](https://core.ac.uk/download/pdf/20641325.pdf) |

The purpose of this crate is to faciliate dataset loading and make it as simple as possible. Loaded datasets are returned as a 
[`linfa::Dataset`](https://docs.rs/linfa/latest/linfa/dataset/type.Dataset.html) structure with named features. The crate also includes helper functions for reading arrays from CSV files.

Additionally, this crate provides utility functions to randomly generate test datasets.

## Using a dataset

To use one of the provided datasets in your project add the `linfa-datasets` crate to your `Cargo.toml` and enable the corresponding feature:
```ignore
linfa-datasets = { version = "0.x", features = ["winequality"] }
```
You can then use the dataset in your working code:
```rust,ignore
let (train, valid) = linfa_datasets::winequality()
    .split_with_ratio(0.8);
```

## Reading from a file

`linfa-datasets` is also capable of reading 2D arrays from CSV files:
```rust,no_run
use std::fs::File;
use std::io::Read;

let file = File::open("data.csv.gz").unwrap();
// Read the array from a GZipped CSV file with a header and separated by commas
let array = linfa_datasets::array_from_gz_csv(file, true, b',').unwrap();
```

## Data generation

To generate datasets randomly, enable the `generate` feature on `linfa-datasets`. The API is in the `generate` module of the crate.
