# Clustering

`linfa-clustering` aims to provide pure Rust implementations of popular clustering algorithms.

_Documentation_: [latest](https://docs.rs/linfa-clustering).

## The big picture

`linfa-clustering` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, a wider effort to
bootstrap a toolkit for classical Machine Learning implemented in pure Rust,
kin in spirit to Python's `scikit-learn`.

You can find a roadmap (and a selection of good first issues)
[here](https://github.com/LukeMathWalker/linfa/issues) - contributors are more than welcome!

## Current state

Right now `linfa-clustering` only provides a single algorithm, `K-Means`, with
a couple of helper functions.

Implementation choices, algorithmic details and a tutorial can be found 
[here](https://docs.rs/linfa-clustering/0.1.0/linfa-clustering/struct.KMeans.html).

Check [here](https://github.com/LukeMathWalker/clustering-benchmarks) for extensive benchmarks against `scikit-learn`'s K-means implementation.

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
