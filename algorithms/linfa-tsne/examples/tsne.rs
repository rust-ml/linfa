use linfa::traits::{Fit, Transformer};
use linfa_reduction::Pca;
use linfa_tsne::{Result, TSne};
use std::{io::Write, process::Command};

fn main() -> Result<()> {
    let ds = linfa_datasets::iris();
    let ds = Pca::params(3).whiten(true).fit(&ds).unwrap().transform(ds);

    let ds = TSne::embedding_size(2)
        .perplexity(10.0)
        .approx_threshold(0.1)
        .transform(ds)?;

    let mut f = std::fs::File::create("examples/iris.dat").unwrap();

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
