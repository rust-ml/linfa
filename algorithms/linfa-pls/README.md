# Partial Least Squares

`linfa-pls` provides a pure Rust implementation of the partial least squares algorithm family.

## The Big Picture

`linfa-pls` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-pls` currently provides an implementation of the following methods: 

 - Partial Least Squares

## Examples

There is an usage example in the `examples/` directory. To run it, use:

```bash
$ cargo run --example pls_regression
```

## BLAS/Lapack backend

See [this section](../../README.md#blaslapack-backend) to enable an external BLAS/LAPACK backend.

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.

