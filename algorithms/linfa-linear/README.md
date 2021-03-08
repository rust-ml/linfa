# Linear Models

`linfa-linear` aims to provide pure Rust implementations of popular linear regression algorithms. 

## The Big Picture

`linfa-linear` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-linear` currently provides an implementation of the following regression algorithms: 
- Ordinary Least Squares
- Generalized Linear Models (GLM)

## Examples

There is an usage example in the `examples/` directory. To run, use:

```bash
$ cargo run --features openblas --example diabetes
$ cargo run --example glm
```

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.
