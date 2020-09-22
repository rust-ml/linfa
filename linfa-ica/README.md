# Independent Component Analysis (ICA)

`linfa-ica` aims to provide pure Rust implementations of ICA algorithms. 

## The Big Picture

`linfa-ica` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-ica` currently provides an implementation of the following factorization methods: 

- Fast Independent Component Analysis (FastICA)

## Examples

There is an usage example in the `examples/` directory. To run, use:

$ cargo run --release --example fast_ica

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.
