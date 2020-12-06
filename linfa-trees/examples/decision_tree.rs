use std::fs::File;
use std::io::Write;

use ndarray::{array, stack, Array, Array1, Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};

fn generate_blobs(means: &[(f64, f64)], samples: usize, mut rng: &mut Isaac64Rng) -> Array2<f64> {
    let out = means
        .into_iter()
        .map(|mean| {
            Array::random_using((samples, 2), StandardNormal, &mut rng) + array![mean.0, mean.1]
        })
        .collect::<Vec<_>>();
    let out2 = out.iter().map(|x| x.view()).collect::<Vec<_>>();

    stack(Axis(0), &out2).unwrap()
}

fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Isaac64Rng::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let n_classes: usize = 4;
    let n = 300;

    println!("Generating training data");

    let train_x = generate_blobs(&[(0., 0.), (1., 4.), (-5., 0.), (4., 4.)], n, &mut rng);
    let train_y = (0..n_classes)
        .map(|x| std::iter::repeat(x).take(n).collect::<Vec<_>>())
        .flatten()
        .collect::<Array1<_>>();

    let dataset = Dataset::new(train_x, train_y).shuffle(&mut rng);
    let (train, test) = dataset.split_with_ratio(0.9);

    println!("Training model with Gini criterion ...");
    let gini_model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(10.0)
        .min_weight_leaf(10.0)
        .fit(&train);

    let gini_pred_y = gini_model.predict(test.records().view());
    let cm = gini_pred_y.confusion_matrix(&test);

    println!("{:?}", cm);

    println!(
        "Test accuracy with Gini criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    println!("Training model with entropy criterion ...");
    let entropy_model = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(100))
        .min_weight_split(10.0)
        .min_weight_leaf(10.0)
        .fit(&train);

    let entropy_pred_y = gini_model.predict(test.records().view());
    let cm = entropy_pred_y.confusion_matrix(&test);

    println!("{:?}", cm);

    println!(
        "Test accuracy with Entropy criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    let feats = entropy_model.features();
    println!("Features trained in this tree {:?}", feats);

    let mut tikz = File::create("decision_tree_example.tex").unwrap();
    tikz.write(gini_model.export_to_tikz().to_string().as_bytes())
        .unwrap();
    println!(" => generate tree description with `latex decision_tree_example.tex`!");
}
