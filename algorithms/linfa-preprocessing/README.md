# Preprocessing
## The Big Picture

`linfa-preprocessing` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state
`linfa-preprocessing` provides a pure Rust implementation of:
* Standard scaling 
* Min-max scaling
* Max Abs Scaling
* Normalization
* Count vectorization
* TfIdf vectorization
* Whitening

## Examples

There are various usage examples in the `examples/` directory. To run, use:

```bash
$ cargo run --release --example count_vectorization
```
```bash
$ cargo run --release --example tfidf_vectorization
```
```bash
$ cargo run --release --example scaling
```
```bash
$ cargo run --release --example whitening
```

## BLAS/Lapack backend

See [this section](../../README.md#blaslapack-backend) to enable an external BLAS/LAPACK backend.

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.
