use std::{error::Error, time::Instant};

use linfa::prelude::*;
use linfa_reduction::random_projection::GaussianRandomProjection;
use linfa_trees::{DecisionTree, SplitQuality};

use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{Array1, Array2};
use rand::thread_rng;

/// Train a Decision tree on the MNIST data set, with and without dimensionality reduction.
fn main() -> Result<(), Box<dyn Error>> {
    // Parameters
    let train_sz = 10_000usize;
    let test_sz = 1_000usize;
    let reduced_dim = 100;

    let NormalizedMnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(train_sz as u32)
        .test_set_length(test_sz as u32)
        .download_and_extract()
        .finalize()
        .normalize();

    let train_data = Array2::from_shape_vec((train_sz, 28 * 28), trn_img)?;
    let train_labels: Array1<usize> =
        Array1::from_shape_vec(train_sz, trn_lbl)?.map(|x| *x as usize);
    let train_dataset = Dataset::new(train_data, train_labels);

    let test_data = Array2::from_shape_vec((test_sz, 28 * 28), tst_img)?;
    let test_labels: Array1<usize> = Array1::from_shape_vec(test_sz, tst_lbl)?.map(|x| *x as usize);

    let params = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(10));

    println!("Training non-reduced model...");
    let start = Instant::now();
    let model: DecisionTree<f32, usize> = params.fit(&train_dataset)?;

    let end = start.elapsed();
    let pred_y = model.predict(&test_data);
    let cm = pred_y.confusion_matrix(&test_labels)?;
    println!("Non-reduced model precision: {}%", 100.0 * cm.accuracy());
    println!("Training time: {:.2}s\n", end.as_secs_f32());

    println!("Training reduced model...");
    let start = Instant::now();
    // Compute the random projection and train the model on the reduced dataset.
    let rng = thread_rng();
    let proj = GaussianRandomProjection::<f32>::params()
        .target_dim(reduced_dim)
        .with_rng(rng)
        .fit(&train_dataset)?;
    let reduced_train_ds = proj.transform(&train_dataset);
    let reduced_test_data = proj.transform(&test_data);
    let model_reduced: DecisionTree<f32, usize> = params.fit(&reduced_train_ds)?;

    let end = start.elapsed();
    let pred_reduced = model_reduced.predict(&reduced_test_data);
    let cm_reduced = pred_reduced.confusion_matrix(&test_labels)?;
    println!(
        "Reduced model precision: {}%",
        100.0 * cm_reduced.accuracy()
    );
    println!("Reduction + training time: {:.2}s", end.as_secs_f32());

    Ok(())
}
