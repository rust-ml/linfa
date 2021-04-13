use encoding::all::ISO_8859_1;
use encoding::DecoderTrap::Strict;
use flate2::read::GzDecoder;
use linfa::metrics::ToConfusionMatrix;
use linfa::traits::{Fit, Predict};
use linfa_bayes::GaussianNbParams;
use linfa_preprocessing::count_vectorization::CountVectorizer;
use ndarray::Array1;
use std::collections::HashSet;
use std::path::Path;
use tar::Archive;

async fn download_20news_bydate() {
    let target = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz";
    let response = reqwest::get(target).await.unwrap();
    let content = response.bytes().await.unwrap().to_vec();
    let tar = GzDecoder::new(content.as_slice());
    let mut archive = Archive::new(tar);
    archive.unpack("./20news/").unwrap();
}

fn load_set(
    path: &'static str,
    desired_targets: &[&str],
) -> Result<(Vec<std::path::PathBuf>, Array1<usize>, usize), std::io::Error> {
    let mut file_paths = Vec::new();
    let mut targets = Vec::new();
    let desired_targets: HashSet<String> = desired_targets.iter().map(|s| s.to_string()).collect();
    let mut ntargets = 0;
    let path = Path::new(path);
    let dir_content = std::fs::read_dir(path)?;
    for sub_dir in dir_content {
        let sub_dir = sub_dir?;
        if sub_dir.file_type().unwrap().is_dir() {
            let dir_name = sub_dir.file_name().into_string().unwrap();
            if !desired_targets.is_empty() && !desired_targets.contains(&dir_name) {
                continue;
            }
            for sub_file in std::fs::read_dir(sub_dir.path())? {
                let sub_file = sub_file?;
                if sub_file.file_type().unwrap().is_file() {
                    file_paths.push(sub_file.path());
                }
                // each directory is a target
                targets.push(ntargets);
            }
            ntargets = ntargets + 1;
        }
    }
    let targets = Array1::from_shape_vec(targets.len(), targets).unwrap();
    Ok((file_paths, targets, ntargets))
}

fn load_train_set(
    desired_targets: &[&str],
) -> Result<(Vec<std::path::PathBuf>, Array1<usize>, usize), std::io::Error> {
    load_set("./20news/20news-bydate-train", desired_targets)
}

fn load_test_set(
    desired_targets: &[&str],
) -> Result<(Vec<std::path::PathBuf>, Array1<usize>, usize), std::io::Error> {
    load_set("./20news/20news-bydate-test", desired_targets)
}

fn delete_20news_bydate() {
    std::fs::remove_dir_all("./20news").unwrap();
}

#[tokio::main]
async fn main() {
    println!("Let's download and unpack the \"20 news\" text classification dataset first");
    download_20news_bydate().await;

    // Restrict possible targets to get a simpler problem. The full dataset has 20 targets total
    let desired_targets = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ];

    // --- Training set

    println!("Now let's load the training set along with the target for each document");
    let (training_filenames, training_targets, ntargets) =
        load_train_set(&desired_targets).unwrap();

    println!();
    println!(
        "The training set is comprised of {} documents with {} different targets",
        training_filenames.len(),
        ntargets
    );

    println!();
    println!("Now let's try to fit a Tf-Idf vectorizer on this set without any restrictions");

    // The 20news dataset's documents are stored in Latin1 encoding so we need to set the equivalent ISO
    // code as the encoding. Using a strict trap will stop the fitting, raising an error, if a file containing an invalid text sequence
    // is encountered
    let vectorizer = CountVectorizer::default()
        .fit_files(&training_filenames, ISO_8859_1, Strict)
        .unwrap();
    println!(
        "We obtain a vocabulary with {} entries",
        vectorizer.nentries()
    );

    println!();
    println!(
        "Now let's generate a matrix containing the tf-idf value of each entry in each document"
    );
    // Transforming gives a sparse dataset, we make it dense in order to be able to fit the Naive Bayes model
    let training_records = vectorizer
        .transform_files(&training_filenames, ISO_8859_1, Strict)
        .to_dense();
    // Currently linfa only allows real valued features so we have to transform the integer counts to floats
    let training_records = training_records.mapv(|c| c as f32);

    println!(
        "We obtain a {}x{} matrix of counts for the vocabulary entries",
        training_records.dim().0,
        training_records.dim().1
    );

    // let's transform the records into a dataset with no targets
    let training_dataset = (training_records, training_targets).into();

    println!();
    println!("Let's try to fit a Naive Bayes model to this set");
    let model = GaussianNbParams::params().fit(&training_dataset).unwrap();
    let training_prediction = model.predict(&training_dataset);

    let cm = training_prediction
        .confusion_matrix(&training_dataset)
        .unwrap();
    // 0.9944
    let accuracy = cm.f1_score();
    println!("The fitted model has a training f1 score of {}", accuracy);

    // --- Test set

    println!();
    println!("Let's try the accuracy on the test set now");
    let (test_filenames, test_targets, ntargets) = load_test_set(&desired_targets).unwrap();

    println!(
        "The test set is comprised of {} documents with {} different targets",
        test_filenames.len(),
        ntargets
    );
    let test_records = vectorizer
        .transform_files(&test_filenames, ISO_8859_1, Strict)
        .to_dense();
    let test_records = test_records.mapv(|c| c as f32);
    let test_dataset = (test_records, test_targets).into();
    // Let's predict the test data targets
    let test_prediction: Array1<usize> = model.predict(&test_dataset);
    let cm = test_prediction.confusion_matrix(&test_dataset).unwrap();
    // 0.9523
    let accuracy = cm.f1_score();
    println!("The model has a test f1 score of {}", accuracy);

    delete_20news_bydate();
}
