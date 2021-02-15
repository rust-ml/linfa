# Examples

## FastICA

```
cargo run --example fast_ica
```

This example creates three .npy files, we plot them using python's [matplotlib](https://matplotlib.org/) separately.

ICA algorithms do not retain the ordering or the sign of the input, hence they can differ in the output.

![fast_ica_example_plot](images/fast_ica.png)
