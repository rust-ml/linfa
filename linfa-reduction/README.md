# Dimensional Reduction

`linfa-reduction` aims to provide pure Rust implementations of dimensional reduction algorithms. 

## The Big Picture

`linfa-reduction` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-reduction` currently provides an implementation of the following dimensional reduction methods: 
- Diffusion Mapping
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)

## Examples

There is an usage example in the `examples/` directory. To run, use:

```bash
$ cargo run --release --example diffusion_map
$ cargo run --release --example pca
$ cargo run --release --example fast_ica
```

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.
