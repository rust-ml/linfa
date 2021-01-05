use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa::Dataset;
use ndarray::prelude::*;
use ndarray_csv::Array2Reader;

#[cfg(any(feature = "iris", feature = "diabetes", feature = "winequality"))]
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
/// Read in the iris-flower dataset from dataset path
/// The `.csv` data is two dimensional: Axis(0) denotes y-axis (rows), Axis(1) denotes x-axis (columns)
pub fn iris() -> Dataset<f64, usize> {
    let data = include_bytes!("../data/iris.csv.gz");
    let array = array_from_buf(&data[..]);

    let (data, targets) = (
        array.slice(s![.., 0..4]).to_owned(),
        array.column(4).to_owned(),
    );

    Dataset::new(data, targets).map_targets(|x| *x as usize)
}

#[cfg(feature = "diabetes")]
pub fn diabetes() -> Dataset<f64, f64> {
    let data = include_bytes!("../data/diabetes_data.csv.gz");
    let data = array_from_buf(&data[..]);

    let targets = include_bytes!("../data/diabetes_target.csv.gz");
    let targets = array_from_buf(&targets[..]).column(0).to_owned();

    Dataset::new(data, targets)
}

#[cfg(feature = "winequality")]
pub fn winequality() -> Dataset<f64, usize> {
    let data = include_bytes!("../data/winequality-red.csv.gz");
    let array = array_from_buf(&data[..]);

    let (data, targets) = (
        array.slice(s![.., 0..11]).to_owned(),
        array.column(11).to_owned(),
    );

    Dataset::new(data, targets).map_targets(|x| *x as usize)
}
