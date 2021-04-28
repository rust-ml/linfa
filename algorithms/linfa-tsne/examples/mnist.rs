use linfa::traits::{Fit, Transformer};
use linfa_reduction::Pca;
use linfa_tsne::{Result, TSne};
use mnist::{Mnist, MnistBuilder};
use std::{io::Write, process::Command};
use ndarray::Array;
use linfa::Dataset;

fn main() -> Result<()> {
    let (trn_size, rows, cols) = (50_000usize, 28, 28);

    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .download_and_extract()
        .finalize();

    let trn_img = Array::from_shape_vec((trn_size, rows * cols), trn_img)?
        .mapv(|x| (x as f64) / 255.);

    let trn_lbl = Array::from_shape_vec((trn_size, 1), trn_lbl)?;
    let ds = Dataset::new(trn_img, trn_lbl);

    let ds = Pca::params(50).whiten(false).fit(&ds).transform(ds);

    println!("Transformed to 10 dimensions and whitened");

    let ds = TSne::embedding_size(2)
        .perplexity(50.0)
        .approx_threshold(0.5)
        .max_iter(1000)
        .transform(ds)?;

    let mut f = std::fs::File::create("examples/mnist.dat").unwrap();

    for (x, y) in ds.sample_iter() {
        f.write(format!("{} {} {}\n", x[0], x[1], y[0]).as_bytes())
            .unwrap();
    }

    Command::new("gnuplot")
        .arg("-p")
        .arg("examples/iris_plot.plt")
        .spawn()
        .expect(
            "Failed to launch gnuplot. Pleasure ensure that gnuplot is installed and on the $PATH.",
        );

    Ok(())
}
