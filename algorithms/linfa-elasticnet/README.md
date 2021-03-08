# Elastic Net

`linfa-elasticnet` aims to provide pure Rust implementations of elastic net linear regression.

## The Big Picture

`linfa-elasticnet` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

The `linfa-elasticnet` crate provides linear regression with ridge and LASSO constraints. The solver uses coordinate descent to find an optimal solution.

## Examples

There is an usage example in the `examples/` directory. To run, use:

```bash
$ cargo run --features linfa/intel-mkl-system --example elasticnet
```

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.
