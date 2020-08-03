# linfa

[![Build Status](https://travis-ci.org/rust-ml/linfa.svg?branch=master)](https://travis-ci.org/rust-ml/linfa)

> _**linfa**_ (Italian) / _**sap**_ (English):
> 
> The **vital** circulating fluid of a plant.


`linfa` aims to provide a comprehensive toolkit to build Machine Learning applications
with Rust.

Kin in spirit to Python's `scikit-learn`, it focuses on common preprocessing tasks
and classical ML algorithms for your everyday ML tasks.

_Documentation_: [latest](https://docs.rs/linfa)
_Community chat_: [Gitter](https://gitter.im/linfa-ml/community)

## Current state

Such bold ambitions! Where are we now? [Are we learning yet?](http://www.arewelearningyet.com/)

Not really: `linfa` only provides a single algorithm, [`K-Means`](clustering/struct.KMeans.html),
with a couple of helper functions.

There is a long way to go to fulfill its bold mission statement, but there is [significant](https://github.com/rust-ml/discussion/issues/1)
[lurking](https://github.com/rust-lang/wg-governance/issues/11) [interest](https://www.reddit.com/r/rust/comments/dvcvo7/rust_2020_scientific_rust/) in the Rust ecosystem when it comes to ML and its surroundings:
sometimes a small spark is all you need to light a beacon fire.

In fact, it is a [firm belief of mine](https://www.youtube.com/watch?v=odI_LY8AIqo&t=8s) that only a significant community effort can nurture,
build and sustain an ML ecosystem in Rust - there is no other way forward.

Even this humble beginning, the [`K-Means` algorithm](clustering/struct.KMeans.html), is the result of [a community workshop](https://github.com/LukeMathWalker/ndarray-koans) at RustFest 2019,
with a bunch of different people chipping in to provide [Python bindings](https://github.com/LukeMathWalker/linfa-python) and interesting
[performance benchmarks](https://github.com/LukeMathWalker/clustering-benchmarks).

We just need to keep walking down the same path.

If this strikes a chord with you, please take a look at the [roadmap](https://github.com/LukeMathWalker/linfa/issues)
and get involved!

# License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.
