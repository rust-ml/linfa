use curl::easy::Easy;
use flate2::read::GzDecoder;
use linfa_preprocessing::tf_idf_vectorization::TfIdfVectorizer;
use linfa_preprocessing::CountVectorizer;
use std::path::Path;
use tar::Archive;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::benchmarks::config;

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

fn fit_vectorizer(file_names: &Vec<std::path::PathBuf>) {
    CountVectorizer::params()
        .document_frequency(0.05, 0.75)
        .n_gram_range(1, 2)
        .fit_files(
            file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap();
}

fn fit_tf_idf(file_names: &Vec<std::path::PathBuf>) {
    TfIdfVectorizer::default()
        .document_frequency(0.05, 0.75)
        .n_gram_range(1, 2)
        .fit_files(
            file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap();
}

fn fit_transform_vectorizer(file_names: &Vec<std::path::PathBuf>) {
    CountVectorizer::params()
        .document_frequency(0.05, 0.75)
        .n_gram_range(1, 2)
        .fit_files(
            file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap()
        .transform_files(
            file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        );
}
fn fit_transform_tf_idf(file_names: &Vec<std::path::PathBuf>) {
    TfIdfVectorizer::default()
        .document_frequency(0.05, 0.75)
        .n_gram_range(1, 2)
        .fit_files(
            file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        )
        .unwrap()
        .transform_files(
            file_names,
            encoding::all::ISO_8859_1,
            encoding::DecoderTrap::Strict,
        );
}

fn bench(c: &mut Criterion) {
    let mut benchmark = c.benchmark_group("Linfa_preprocessing_vectorizer");
    config::set_default_benchmark_configs(&mut benchmark);

    let file_names = load_20news_bydate();

    benchmark.bench_function(
        BenchmarkId::new("Fit-Vectorizer", "20news_bydate"),
        |bencher| {
            bencher.iter(|| {
                fit_vectorizer(black_box(&file_names));
            });
        },
    );

    benchmark.bench_function(BenchmarkId::new("Fit-Tf-Idf", "20news_bydate"), |bencher| {
        bencher.iter(|| {
            fit_tf_idf(black_box(&file_names));
        });
    });

    benchmark.bench_function(
        BenchmarkId::new("Fit-Transfor-Vectorizer", "20news_bydate"),
        |bencher| {
            bencher.iter(|| {
                fit_transform_vectorizer(black_box(&file_names));
            });
        },
    );

    benchmark.bench_function(
        BenchmarkId::new("Fit-Transfor-Tf-Idf", "20news_bydate"),
        |bencher| {
            bencher.iter(|| {
                fit_transform_tf_idf(black_box(&file_names));
            });
        },
    );
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, bench);

criterion_main!(benches);
