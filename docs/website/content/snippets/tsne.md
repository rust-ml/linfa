+++
title = "Barnes-Hut t-SNE"
+++
```rust
// normalize the iris dataset
let ds = linfa_datasets::iris();
let ds = Pca::params(3).whiten(true).fit(&ds).transform(ds);

// transform to two-dimensional embeddings
let ds = TSne::embedding_size(2)
    .perplexity(10.0)
    .approx_threshold(0.1)
    .transform(ds)?;

// write embedding to file
let mut f = File::create("iris.dat")?;
for (x, y) in ds.sample_iter() {
    f.write(format!("{} {} {}\n", x[0], x[1], y[0]).as_bytes())?;
}
```
