use std::io::Read;

use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa::Dataset;
use ndarray::prelude::*;
use ndarray_csv::{Array2Reader, ReadError};

/// Convert Gzipped CSV bytes into 2D array
pub fn array_from_gz_csv<R: Read>(
    gz: R,
    has_headers: bool,
    separator: u8,
) -> Result<Array2<f64>, ReadError> {
    // unzip file
    let file = GzDecoder::new(gz);
    array_from_csv(file, has_headers, separator)
}

/// Convert CSV bytes into 2D array
pub fn array_from_csv<R: Read>(
    csv: R,
    has_headers: bool,
    separator: u8,
) -> Result<Array2<f64>, ReadError> {
    // parse CSV
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .delimiter(separator)
        .from_reader(csv);

    // extract ndarray
    reader.deserialize_array2_dynamic()
}

#[cfg(feature = "iris")]
/// Read in the iris-flower dataset from dataset path.
// The `.csv` data is two dimensional: Axis(0) denotes y-axis (rows), Axis(1) denotes x-axis (columns)
pub fn iris() -> Dataset<f64, usize, Ix1> {
    let data = include_bytes!("../data/iris.csv.gz");
    let array = array_from_gz_csv(&data[..], true, b',').unwrap();

    let (data, targets) = (
        array.slice(s![.., 0..4]).to_owned(),
        array.column(4).to_owned(),
    );

    let feature_names = vec!["sepal length", "sepal width", "petal length", "petal width"];

    Dataset::new(data, targets)
        .map_targets(|x| *x as usize)
        .with_feature_names(feature_names)
}

#[cfg(feature = "mnist")]
/// Read in the mnist_dataset from dataset path.
pub fn mnist() -> (Dataset<f64, usize, Ix1>, Dataset<f64, usize, Ix1>) {
    let train_data = include_bytes!("../data/mnist_train.csv.gz");
    let test_data = include_bytes!("../data/mnist_test.csv.gz");
    let train_array = array_from_gz_csv(&train_data[..], true, b',').unwrap();
    let test_array = array_from_gz_csv(&test_data[..], true, b',').unwrap();
    let (data, targets) = (
        train_array.slice(s![.., 1..]).to_owned(),
        train_array.column(0).to_owned(),
    );
    let train = Dataset::new(data, targets).map_targets(|x| *x as usize);
    let (data, targets) = (
        test_array.slice(s![.., 1..]).to_owned(),
        test_array.column(0).to_owned(),
    );

    let test = Dataset::new(data, targets).map_targets(|x| *x as usize);
    (train, test)
}

#[cfg(feature = "boston")]
/// Read in the Boston housing dataset from dataset path.
pub fn boston() -> Dataset<f64, f64, Ix1> {
    let data = include_bytes!("../data/BostonHousing.csv.gz");
    let array = array_from_gz_csv(&data[..], true, b',').unwrap();

    // Assuming that the last column is the target (e.g., median house value)
    let (data, targets) = (
        array.slice(s![.., ..-1]).to_owned(),  // All columns except the last
        array.column(array.ncols() - 1).to_owned(),  // Last column as target
    );

    let feature_names = vec![
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ];

    Dataset::new(data, targets)
        .with_feature_names(feature_names)
}

#[cfg(feature = "diabetes")]
/// Read in the diabetes dataset from dataset path
pub fn diabetes() -> Dataset<f64, f64, Ix1> {
    let data = include_bytes!("../data/diabetes_data.csv.gz");
    let data = array_from_gz_csv(&data[..], true, b',').unwrap();

    let targets = include_bytes!("../data/diabetes_target.csv.gz");
    let targets = array_from_gz_csv(&targets[..], true, b',')
        .unwrap()
        .column(0)
        .to_owned();

    let feature_names = vec![
        "age",
        "sex",
        "body mass index",
        "blood pressure",
        "t-cells",
        "low-density lipoproteins",
        "high-density lipoproteins",
        "thyroid stimulating hormone",
        "lamotrigine",
        "blood sugar level",
    ];

    Dataset::new(data, targets).with_feature_names(feature_names)
}

#[cfg(feature = "winequality")]
/// Read in the winequality dataset from dataset path
pub fn winequality() -> Dataset<f64, usize, Ix1> {
    let data = include_bytes!("../data/winequality-red.csv.gz");
    let array = array_from_gz_csv(&data[..], true, b',').unwrap();

    let (data, targets) = (
        array.slice(s![.., 0..11]).to_owned(),
        array.column(11).to_owned(),
    );

    let feature_names = vec![
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ];

    Dataset::new(data, targets)
        .map_targets(|x| *x as usize)
        .with_feature_names(feature_names)
}

#[cfg(feature = "linnerud")]
/// Read in the physical exercise dataset from dataset path.
///
/// Linnerud dataset contains 20 samples collected from 20 middle-aged men in a fitness club.
///
/// ## Features:
/// 3 exercises measurements: Chins, Situps, Jumps
///
/// ## Targets:
/// 3 physiological measurements: Weight, Waist, Pulse
///
/// # Reference:
/// Tenenhaus (1998). La regression PLS: theorie et pratique. Paris: Editions Technip. Table p 15.
pub fn linnerud() -> Dataset<f64, f64> {
    let input_data = include_bytes!("../data/linnerud_exercise.csv.gz");
    let input_array = array_from_gz_csv(&input_data[..], true, b',').unwrap();

    let output_data = include_bytes!("../data/linnerud_physiological.csv.gz");
    let output_array = array_from_gz_csv(&output_data[..], true, b',').unwrap();

    let feature_names = vec!["Chins", "Situps", "Jumps"];

    Dataset::new(input_array, output_array).with_feature_names(feature_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::prelude::*;

    #[cfg(feature = "boston")]
    #[test]
    fn test_boston() {
        let ds = boston();

        assert_eq!(ds.nsamples(), 506);  // Total samples in the dataset
        assert_eq!(ds.nfeatures(), 13);  // Total number of features
        assert_eq!(ds.ntargets(), 1);    // One target variable

        // Optionally, verify the correct feature names are loaded
        let expected_feature_names = vec![
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
        ];
        assert_eq!(ds.feature_names(), expected_feature_names);
    }

    #[cfg(feature = "iris")]
    #[test]
    fn test_iris() {
        let ds = iris();

        // check that we have the right amount of data
        assert_eq!((ds.nsamples(), ds.nfeatures(), ds.ntargets()), (150, 4, 1));

        // check for feature names
        assert_eq!(
            ds.feature_names(),
            &["sepal length", "sepal width", "petal length", "petal width"]
        );

        // check label frequency
        assert_abs_diff_eq!(
            ds.label_frequencies()
                .into_iter()
                .map(|b| b.1)
                .collect::<Array1<_>>(),
            array![50., 50., 50.]
        );

        // perform correlation analysis and assert that petal length and width are correlated
        let _pcc = ds.pearson_correlation_with_p_value(100);
        // TODO: wait for pearson correlation to accept rng
        // assert_abs_diff_eq!(pcc.get_p_values().unwrap()[5], 0.04, epsilon = 0.04);

        // get the mean per feature
        let mean_features = ds.records().mean_axis(Axis(0)).unwrap();
        assert_abs_diff_eq!(
            mean_features,
            array![5.84, 3.05, 3.75, 1.20],
            epsilon = 0.01
        );
    }

    #[cfg(feature = "mnist")]
    #[test]
    fn test_mnist() {
        let (train, test) = mnist();
        assert_eq!(
            (train.nsamples(), train.nfeatures(), train.ntargets()),
            (60000, 784, 1)
        );
        assert_eq!(
            (test.nsamples(), test.nfeatures(), test.ntargets()),
            (10000, 784, 1)
        );
    }

    #[cfg(feature = "diabetes")]
    #[test]
    fn test_diabetes() {
        let ds = diabetes();

        // check that we have the right amount of data
        assert_eq!((ds.nsamples(), ds.nfeatures(), ds.ntargets()), (441, 10, 1));

        // perform correlation analysis and assert that T-Cells and low-density lipoproteins are
        // correlated
        let _pcc = ds.pearson_correlation_with_p_value(100);
        //assert_abs_diff_eq!(pcc.get_p_values().unwrap()[30], 0.02, epsilon = 0.02);

        // get the mean per feature, the data should be normalized
        let mean_features = ds.records().mean_axis(Axis(0)).unwrap();
        assert_abs_diff_eq!(mean_features, Array1::zeros(10), epsilon = 0.005);
    }

    #[cfg(feature = "winequality")]
    #[test]
    fn test_winequality() {
        use approx::abs_diff_eq;

        let ds = winequality();

        // check that we have the right amount of data
        assert_eq!(
            (ds.nsamples(), ds.nfeatures(), ds.ntargets()),
            (1599, 11, 1)
        );

        // check for feature names
        let feature_names = vec![
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ];
        assert_eq!(ds.feature_names(), feature_names);

        // check label frequency
        let compare_to = vec![
            (5, 681.0),
            (7, 199.0),
            (6, 638.0),
            (8, 18.0),
            (3, 10.0),
            (4, 53.0),
        ];

        let freqs = ds.label_frequencies();
        assert!(compare_to.into_iter().all(|(key, val)| {
            freqs
                .get(&key)
                .map(|x| abs_diff_eq!(*x, val))
                .unwrap_or(false)
        }));

        // perform correlation analysis and assert that fixed acidity and citric acid are
        // correlated
        let _pcc = ds.pearson_correlation_with_p_value(100);
        //assert_abs_diff_eq!(pcc.get_p_values().unwrap()[1], 0.05, epsilon = 0.05);
    }

    #[cfg(feature = "linnerud")]
    #[test]
    fn test_linnerud() {
        let ds = linnerud();

        // check that we have the right amount of data
        assert_eq!((ds.nsamples(), ds.nfeatures(), ds.ntargets()), (20, 3, 3));

        // check for feature names
        let feature_names = vec!["Chins", "Situps", "Jumps"];
        assert_eq!(ds.feature_names(), feature_names);

        // get the mean per target: Weight, Waist, Pulse
        let mean_targets = ds.targets().mean_axis(Axis(0)).unwrap();
        assert_abs_diff_eq!(mean_targets, array![178.6, 35.4, 56.1]);
    }
}
