use linfa::prelude::*;
use linfa_logistic::error::Result;
use linfa_logistic::LogisticRegression;

fn main() -> Result<()> {
    // Load dataset. Mutability is needed for fast cross validation
    let mut dataset =
        linfa_datasets::winequality().map_targets(|x| if *x > 6 { "good" } else { "bad" });

    // define a sequence of models to compare. In this case the
    // models will differ by the amount of l2 regularization
    let alphas = &[0.1, 1., 10.];
    let models: Vec<_> = alphas
        .iter()
        .map(|alpha| {
            LogisticRegression::default()
                .alpha(*alpha)
                .max_iterations(150)
        })
        .collect();

    // use cross validation to compute the validation accuracy of each model. The
    // accuracy of each model will be averaged across the folds, 5 in this case
    let accuracies = dataset.cross_validate_single(5, &models, |prediction, truth| {
        Ok(prediction.confusion_matrix(truth)?.accuracy())
    })?;

    // display the accuracy of the models along with their regularization coefficient
    for (alpha, accuracy) in alphas.iter().zip(accuracies.iter()) {
        println!("Alpha: {}, accuracy: {} ", alpha, accuracy);
    }

    Ok(())
}
