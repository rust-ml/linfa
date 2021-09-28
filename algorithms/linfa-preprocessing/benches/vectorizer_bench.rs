use curl::easy::Easy;
use flate2::read::GzDecoder;
use linfa_preprocessing::tf_idf_vectorization::TfIdfVectorizer;
use linfa_preprocessing::CountVectorizer;
use std::path::Path;
use tar::Archive;

fn download_20news_bydate() {
    let mut data = Vec::new();
    let mut easy = Easy::new();
    easy.url("http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz")
        .unwrap();
    {
        let mut transfer = easy.transfer();
        transfer
            .write_function(|new_data| {
                data.extend_from_slice(new_data);
                Ok(new_data.len())
            })
            .unwrap();
        transfer.perform().unwrap();
    }
    let tar = GzDecoder::new(data.as_slice());
    let mut archive = Archive::new(tar);
    archive.unpack("./20news/").unwrap();
}

fn load_20news_bydate() -> Vec<std::path::PathBuf> {
    // Let's help cachegrind and select only 4 targets
    let desired_targets = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ];
    let file_paths = load_test_set(&desired_targets);
    if let Ok(paths) = file_paths {
        paths
    } else {
        download_20news_bydate();
        load_test_set(&desired_targets).unwrap()
    }
}

fn load_set(
    path: &'static str,
    desired_targets: &[&str],
) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    let mut file_paths = Vec::new();
    let desired_targets: std::collections::HashSet<String> =
        desired_targets.iter().map(|s| s.to_string()).collect();
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
            }
        }
    }
    Ok(file_paths)
}

fn _load_train_set(desired_targets: &[&str]) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    load_set("./20news/20news-bydate-train", desired_targets)
}

fn load_test_set(desired_targets: &[&str]) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    load_set("./20news/20news-bydate-test", desired_targets)
}

fn iai_fit_vectorizer() {
    let file_names = load_20news_bydate();
    CountVectorizer::params()
        .document_frequency(0.05, 0.75)
        .n_gram_range(1, 2)
        .fit_files(
            &file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap();
}

fn iai_fit_tf_idf() {
    let file_names = load_20news_bydate();
    TfIdfVectorizer::default()
        .document_frequency(0.05, 0.75)
        .n_gram_range(1, 2)
        .fit_files(
            &file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap();
}

fn iai_fit_transform_vectorizer() {
    let file_names = load_20news_bydate();
    CountVectorizer::params()
        .document_frequency(0.05, 0.75)
        .n_gram_range(1, 2)
        .fit_files(
            &file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap()
        .transform_files(
            &file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        );
}
fn iai_fit_transform_tf_idf() {
    let file_names = load_20news_bydate();
    TfIdfVectorizer::default()
        .document_frequency(0.05, 0.75)
        .n_gram_range(1, 2)
        .fit_files(
            &file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap()
        .transform_files(
            &file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        );
}

macro_rules! iai_main {
    ( $( $func_name:ident ),+ $(,)* ) => {
        mod iai_wrappers {
            $(
                pub fn $func_name() {
                    let _ = iai::black_box(super::$func_name());
                }
            )+
        }

        fn main() {
            load_20news_bydate();
            let benchmarks : &[&(&'static str, fn())]= &[

                $(
                    &(stringify!($func_name), iai_wrappers::$func_name),
                )+
            ];

            iai::runner(benchmarks);
            std::fs::remove_dir_all("./20news").unwrap_or(());
        }
    }
}

iai_main!(
    iai_fit_vectorizer,
    iai_fit_transform_vectorizer,
    iai_fit_tf_idf,
    iai_fit_transform_tf_idf
);
