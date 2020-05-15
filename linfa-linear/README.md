# Linear Models

`linfa-linear` aims to provide pure Rust implementations of 
popular linear regression algorithms. 

_Documentation_: [latest](https://docs.rs/linfa-linear).

## The Big Picture

`linfa-linear` is a crate in the [`linfa`](https://crates.io/crates/linfa) 
ecosystem, a wider effort to bootstrap a toolkit for classical 
Machine Learning implemented in pure Rust, kin in spirit to 
Python's `scikit-learn`.

## Current state

Right now `linfa-linear` provides ordinary least squares regression.

## Examples

There is an usage example in the `examples/diabetes.rs` file, to run it
run

```bash
$ cargo run --features openblas --examples diabetes
```

## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
