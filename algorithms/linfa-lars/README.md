# Least Angle Regression

`linfa-lars` aims to provide pure Rust implementation of least-angle regression algorithm. 

## The Big Picture

`linfa-lars` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

The `linfa-lars` crate currently provides an implementation of the Least-angle regression (LARS) algorithm

See also:
 * [Wikipedia on Least Angle Regression](https://en.wikipedia.org/wiki/Least-angle_regression)


## BLAS/Lapack backend

See [this section](../../README.md#blaslapack-backend) to enable an external BLAS/LAPACK backend.

## Examples

There is an usage example in the `examples/` directory. To run, use:

```bash
$ cargo run --example lars
```

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.