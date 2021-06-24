use linfa::traits::{Fit, Transformer};
use linfa::Dataset;
use linfa_reduction::Pca;
use linfa_tsne::{Result, TSne};
use mnist::{Mnist, MnistBuilder};
use ndarray::Array;
use std::{io::Write, process::Command};

fn main() -> Result<()> {
    // use 50k samples from the MNIST dataset
    let (trn_size, rows, cols) = (50_000usize, 28, 28);

    // download and extract it into a dataset
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .download_and_extract()
        .finalize();

    // create a dataset from it
    let ds = Dataset::new(
        Array::from_shape_vec((trn_size, rows * cols), trn_img)?.mapv(|x| (x as f64) / 255.),
        Array::from_shape_vec((trn_size, 1), trn_lbl)?,
    );

    // reduce to 50 dimension without whitening
    let ds = Pca::params(50)
        .whiten(false)
        .fit(&ds)
        .unwrap()
        .transform(ds);

    // calculate a two-dimensional embedding with Barnes-Hut t-SNE
    let ds = TSne::embedding_size(2)
        .perplexity(50.0)
        .approx_threshold(0.5)
        .max_iter(1000)
        .transform(ds)?;

    // write out
    let mut f = std::fs::File::create("examples/mnist.dat").unwrap();

    for (x, y) in ds.sample_iter() {
        f.write_all(format!("{} {} {}\n", x[0], x[1], y[0]).as_bytes())
            .unwrap();
    }

    // and plot with gnuplot
    Command::new("gnuplot")
        .arg("-p")
        .arg("examples/mnist_plot.plt")
        .spawn()
        .expect(
            "Failed to launch gnuplot. Pleasure ensure that gnuplot is installed and on the $PATH.",
        );

    Ok(())
}
