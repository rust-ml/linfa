//! Logistic regression on sparse feature matrices.
//!
//! Text features such as TF-IDF are naturally sparse: each document only contains a handful of
//! the words in the vocabulary. `linfa-preprocessing` returns them as a `sprs::CsMat`, which
//! `linfa-logistic` can fit and predict on directly, without converting to a dense array.

use linfa::prelude::*;
use linfa_logistic::{LogisticRegression, MultiLogisticRegression};
use linfa_preprocessing::tf_idf_vectorization::TfIdfVectorizer;
use ndarray::{array, Array1};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use sprs::{CsMat, TriMat};
use std::time::Instant;

const TOPICS: [&str; 3] = ["sports", "cooking", "astronomy"];

fn main() {
    text_classification();
    high_dimensional();
}

fn text_classification() {
    println!("== Part 1: topic classification from TF-IDF features\n");

    let train = array![
        "the striker scored a late goal and the crowd cheered",
        "the goalkeeper saved a penalty in the final minute",
        "our team won the league after a tense away match",
        "the coach praised the defence after the cup match",
        "a hat trick from the striker sealed the victory",
        "the referee showed a red card to the midfielder",
        "fans travelled to the stadium for the derby",
        "the runner set a new record at the athletics meeting",
        "simmer the tomato sauce with garlic and fresh basil",
        "knead the dough and let the bread rise overnight",
        "season the steak with salt and pepper before grilling",
        "whisk the eggs with sugar until the batter is pale",
        "roast the vegetables in the oven with olive oil",
        "stir the risotto slowly and add the stock gradually",
        "bake the cake at a moderate oven temperature",
        "chop the onions and fry them in butter until golden",
        "the telescope captured a faint distant galaxy",
        "astronomers detected a new planet orbiting the star",
        "the comet will be visible in the night sky next week",
        "a supernova explosion briefly outshone its whole galaxy",
        "the moon passed in front of the sun during the eclipse",
        "the orbit of the planet around its star is elliptical",
        "the observatory measured light from a distant quasar",
        "jupiter and saturn appear close together in the night sky",
    ];
    // Eight sentences per topic, in the order of `TOPICS`.
    let train_topics = Array1::from_shape_fn(train.len(), |i| i / 8);

    let test = array![
        "the striker missed a penalty but the team still won",
        "fry the garlic in olive oil and add the tomato",
        "a new comet was spotted by the observatory telescope",
        "the crowd sang as the goalkeeper lifted the cup",
        "let the dough rest before you bake the bread",
        "the galaxy contains billions of stars and planets",
    ];
    let test_topics = array![0, 1, 2, 0, 1, 2];

    let vectorizer = TfIdfVectorizer::default().fit(&train).unwrap();
    let x_train: CsMat<f64> = vectorizer.transform(&train).unwrap();
    let x_test: CsMat<f64> = vectorizer.transform(&test).unwrap();
    println!(
        "TF-IDF matrix: {} documents x {} terms, {} stored values ({:.1}% dense)",
        x_train.rows(),
        x_train.cols(),
        x_train.nnz(),
        100.0 * x_train.density()
    );

    let train_set = DatasetBase::new(x_train.clone(), train_topics.clone());
    let model = MultiLogisticRegression::default()
        .alpha(0.1)
        .fit(&train_set)
        .unwrap();

    let predicted = model.predict(&x_test);
    let probabilities = model.predict_probabilities(&x_test);
    println!("\nPredictions on unseen sentences:");
    for ((sentence, &topic), probs) in test.iter().zip(predicted.iter()).zip(probabilities.rows()) {
        println!(
            "  {:<10} (p = {:.2})  \"{sentence}\"",
            TOPICS[topic], probs[topic]
        );
    }
    let correct = predicted
        .iter()
        .zip(test_topics.iter())
        .filter(|(p, t)| p == t)
        .count();
    println!("Accuracy: {correct}/{}", test_topics.len());

    // The sparse fit solves exactly the same problem as a fit on the densified matrix.
    let dense_model = MultiLogisticRegression::default()
        .alpha(0.1)
        .fit(&Dataset::new(x_train.to_dense(), train_topics))
        .unwrap();
    let max_param_diff = (model.params() - dense_model.params())
        .mapv(f64::abs)
        .fold(0.0, |a: f64, &b| a.max(b));
    let max_prob_diff = (&probabilities - &dense_model.predict_probabilities(&x_test.to_dense()))
        .mapv(f64::abs)
        .fold(0.0, |a: f64, &b| a.max(b));
    println!("Largest difference to the dense fit: {max_param_diff:.1e} in parameters, {max_prob_diff:.1e} in probabilities");
    assert!(max_param_diff < 1e-6 && max_prob_diff < 1e-6);
    assert_eq!(predicted, dense_model.predict(&x_test.to_dense()));
}

/// Builds a random binary design matrix in CSR format. Like words in documents, every row has
/// a few "topical" features drawn from the first `n_informative` columns, which carry the
/// signal, plus many rare features drawn from the remaining columns, which are pure noise.
/// Labels are drawn from a logistic model with weights `true_weights`.
fn synthetic_problem(
    rng: &mut Xoshiro256Plus,
    n_samples: usize,
    (n_informative, n_features): (usize, usize),
    (informative_per_row, noise_per_row): (usize, usize),
    true_weights: &Array1<f64>,
) -> (CsMat<f64>, Array1<bool>) {
    let mut triplets = TriMat::new((n_samples, n_features));
    let mut labels = Array1::from_elem(n_samples, false);
    for (row, label) in labels.iter_mut().enumerate() {
        let mut z = 0.0;
        for _ in 0..informative_per_row {
            let col = rng.gen_range(0..n_informative);
            triplets.add_triplet(row, col, 1.0);
            z += true_weights[col];
        }
        for _ in 0..noise_per_row {
            triplets.add_triplet(row, rng.gen_range(n_informative..n_features), 1.0);
        }
        *label = rng.gen::<f64>() < 1.0 / (1.0 + (-z).exp());
    }
    // Duplicated (row, col) pairs are summed when converting to CSR.
    (triplets.to_csr(), labels)
}

fn high_dimensional() {
    println!("\n== Part 2: a 100 000-feature problem, never densified\n");

    let (n_train, n_test) = (20_000, 5_000);
    let (n_informative, n_features) = (1_000, 100_000);
    let nnz_per_row = (10, 30);
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    let true_weights = Array1::from_shape_fn(n_features, |j| {
        if j < n_informative {
            if j % 2 == 0 {
                1.0
            } else {
                -1.0
            }
        } else {
            0.0
        }
    });
    let (x_train, y_train) = synthetic_problem(
        &mut rng,
        n_train,
        (n_informative, n_features),
        nnz_per_row,
        &true_weights,
    );
    let (x_test, y_test) = synthetic_problem(
        &mut rng,
        n_test,
        (n_informative, n_features),
        nnz_per_row,
        &true_weights,
    );

    let sparse_mib = (x_train.nnz() * (8 + 8) + (n_train + 1) * 8) as f64 / (1 << 20) as f64;
    let dense_gib = (n_train * n_features * 8) as f64 / (1 << 30) as f64;
    println!(
        "Training matrix: {n_train} x {n_features}, {} stored values",
        x_train.nnz()
    );
    println!("Memory: {sparse_mib:.1} MiB as CSR, {dense_gib:.1} GiB if densified");

    let start = Instant::now();
    let model = LogisticRegression::default()
        .alpha(10.0)
        .max_iterations(200)
        .fit(&DatasetBase::new(x_train.view(), y_train.clone()))
        .unwrap();
    println!("Fitted in {:.2?}", start.elapsed());

    let accuracy = |x: &CsMat<f64>, y: &Array1<bool>| {
        let predicted = model.predict(x);
        predicted.iter().zip(y).filter(|(p, t)| p == t).count() as f64 / y.len() as f64
    };
    let train_accuracy = accuracy(&x_train, &y_train);
    let test_accuracy = accuracy(&x_test, &y_test);
    println!("Accuracy: {train_accuracy:.3} on training data, {test_accuracy:.3} on held-out data");
    // The labels are noisy draws, so even the planted model is not always right.
    let oracle_accuracy = (&x_test * &true_weights)
        .iter()
        .zip(&y_test)
        .filter(|(&z, &t)| (z > 0.0) == t)
        .count() as f64
        / n_test as f64;
    println!("The planted model itself scores {oracle_accuracy:.3} on the held-out data");

    let sign_matches = (0..n_informative)
        .filter(|&j| model.params()[j].signum() == true_weights[j].signum())
        .count();
    println!(
        "Recovered the sign of {sign_matches}/{n_informative} informative weights; largest weight among the {} noise features: {:.3}",
        n_features - n_informative,
        model
            .params()
            .iter()
            .skip(n_informative)
            .fold(0.0, |a: f64, &b| a.max(b.abs()))
    );
}
