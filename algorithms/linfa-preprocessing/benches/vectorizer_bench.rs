use flate2::read::GzDecoder;
use linfa_preprocessing::count_vectorization::CountVectorizer;
use linfa_preprocessing::tf_idf_vectorization::TfIdfVectorizer;
use std::path::Path;
use tar::Archive;

#[tokio::main]
async fn download_20news_bydate() -> Vec<std::path::PathBuf> {
    let file_paths = load_train_filenames();
    if file_paths.is_err() {
        let target = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz";
        let response = reqwest::get(target).await.unwrap();
        let content = response.bytes().await.unwrap().to_vec();
        let tar = GzDecoder::new(content.as_slice());
        let mut archive = Archive::new(tar);
        archive.unpack(".").unwrap();
        load_train_filenames().unwrap()
    } else {
        file_paths.unwrap()
    }
}

fn load_train_filenames() -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    let mut file_paths = Vec::new();
    let path = Path::new("./20news-bydate-train");
    let dir_content = std::fs::read_dir(path)?;
    for sub_dir in dir_content {
        let sub_dir = sub_dir?;
        if sub_dir.file_type().unwrap().is_dir() {
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

fn load_test_filenames() -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    let mut file_paths = Vec::new();
    let path = Path::new("./20news-bydate-test");
    let dir_content = std::fs::read_dir(path)?;
    for sub_dir in dir_content {
        let sub_dir = sub_dir?;
        if sub_dir.file_type().unwrap().is_dir() {
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

fn iai_benchmark_count_vectorizer() {
    let file_names = load_test_filenames().unwrap();
    let vectorizer = CountVectorizer::default()
        .document_frequency(0.05, 0.5)
        .fit_files(
            iai::black_box(&file_names),
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap();
    let _transformed = vectorizer.transform_files(
        iai::black_box(&file_names),
        encoding::all::ISO_8859_1,
        encoding::DecoderTrap::Strict,
    );
}

fn iai_benchmark_tf_idf() {
    let file_names = load_test_filenames().unwrap();
    let vectorizer = TfIdfVectorizer::default()
        .document_frequency(0.05, 0.5)
        .fit_files(
            &file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap();
    let _transformed = vectorizer.transform_files(
        &file_names,
        encoding::all::ISO_8859_1,
        encoding::DecoderTrap::Strict,
    );
}

macro_rules! main {
    ( $( $func_name:ident ),+ $(,)* ) => {
        mod iai_wrappers {
            $(
                pub fn $func_name() {
                    let _ = iai::black_box(super::$func_name());
                }
            )+
        }

        fn main() {
            download_20news_bydate();
            let benchmarks : &[&(&'static str, fn())]= &[

                $(
                    &(stringify!($func_name), iai_wrappers::$func_name),
                )+
            ];

            iai::runner(benchmarks);
        }
    }
}

main!(iai_benchmark_count_vectorizer, iai_benchmark_tf_idf);
