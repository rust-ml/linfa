# t-SNE

`linfa-tsne` provides a pure Rust implementation of exact and Barnes-Hut t-SNE.

## The Big Picture

`linfa-tsne` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-tsne` currently provides an implementation of the following methods: 

- exact solution t-SNE
- Barnes-Hut t-SNE

It wraps the [bhtsne](https://github.com/frjnn/bhtsne) crate, all kudos to them.

## Examples

There is an usage example in the `examples/` directory. The example uses a BLAS backend, to run it and use the `intel-mkl` library do:

```bash
$ cargo run --example tsne --features linfa/intel-mkl-system
```

You have to install the `gnuplot` library for plotting.

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.

