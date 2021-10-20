# Logistic Regression

## The Big Picture

`linfa-logistic` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state
`linfa-logistic` provides pure Rust implementations of two-class and multinomial logistic regression models.

## Examples
There are usage examples in the `examples/` directory.

To run the two-class example, use:
```bash
$ cargo run --example winequality --features linfa/<blas-library>
```

To run the multinomial example, use:
```bash
$ cargo run --example winequality_multi --features linfa/<blas-library>
```

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.
