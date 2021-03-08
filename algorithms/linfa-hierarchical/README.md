# Clustering

`linfa-hierarchical` provides an implementation of agglomerative hierarchical clustering. 
In this clustering algorithm, each point is first considered as a separate cluster. During each
step, two points are merged into new clusters, until a stopping criterion is reached. The distance
between the points is computed as the negative-log transform of the similarity kernel.

_Documentation_: [latest](https://docs.rs/linfa).

## The big picture

`linfa-hierarchical` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, a wider effort to bootstrap a toolkit for classical Machine Learning implemented in pure Rust, akin in spirit to Python's `scikit-learn`.

## Current state

`linfa-hierarchical` implements agglomerative hierarchical clustering with support of the [kodama](https://docs.rs/kodama/0.2.3/kodama/) crate.

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
