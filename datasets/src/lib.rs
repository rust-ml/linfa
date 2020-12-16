use ndarray::{Array2, Array1, s};
use flate2::read::GzDecoder;
use ndarray_csv::Array2Reader;
use csv::ReaderBuilder;
use linfa::Dataset;

fn array_from_buf(buf: &[u8]) -> Array2<f64> {
    // unzip file
    let file = GzDecoder::new(buf);
    // create a CSV reader with headers and `;` as delimiter
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_reader(file);

    // extract ndarray
    reader.deserialize_array2_dynamic().unwrap()
}

#[cfg(feature = "iris")]
pub fn irisflower() -> Dataset<Array2<f64>, Vec<usize>> {
    let data = include_bytes!("../data/iris.csv.gz");
    let array = array_from_buf(&data[..]);

    let (data, targets) = (
        array.slice(s![.., 0..4]).to_owned(),
        array.column(4).to_owned(),
    );

    Dataset::new(data, targets)
        .map_targets(|x| *x as usize)
}
