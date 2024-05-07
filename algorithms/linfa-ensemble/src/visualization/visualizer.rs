// visualization.rs
use ndarray::Array1;
use plotters::prelude::*;

/// Plots a scatter plot comparing actual and predicted values.
///
/// # Arguments
/// * `actual_train` - Actual values from the training dataset.
/// * `predicted_train` - Predicted values from the training dataset.
/// * `actual_test` - Actual values from the test dataset.
/// * `predicted_test` - Predicted values from the test dataset.
/// * `file_name` - Filename where the plot will be saved.
pub fn plot_scatter(
    actual_train: &Array1<f64>,
    predicted_train: &Array1<f64>,
    actual_test: &Array1<f64>,
    predicted_test: &Array1<f64>,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_actual = actual_train.iter().chain(actual_test.iter()).copied().fold(f64::INFINITY, f64::min);
    let max_actual = actual_train.iter().chain(actual_test.iter()).copied().fold(f64::NEG_INFINITY, f64::max);
    let min_predicted = predicted_train.iter().chain(predicted_test.iter()).copied().fold(f64::INFINITY, f64::min);
    let max_predicted = predicted_train.iter().chain(predicted_test.iter()).copied().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Predicted vs Actual Values", ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_actual..max_actual, min_predicted..max_predicted)?;

    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    chart.draw_series(
        actual_test.iter()
            .zip(predicted_test.iter())
            .map(|(actual, predicted)| Circle::new((*actual, *predicted), 2, RED.filled())),
    )?.label("Predicted Data Points")
     .legend(|(x, y)| Circle::new((x, y), 2, RED.filled()));

    chart.draw_series(
        actual_train.iter()
            .zip(predicted_train.iter())
            .map(|(actual, predicted)| Circle::new((*actual, *predicted), 2, BLUE.filled())),
    )?.label("Actual Data Points")
     .legend(|(x, y)| Circle::new((x, y), 2, BLUE.filled()));

    chart.configure_series_labels().background_style(WHITE).draw()?;

    Ok(())
}

/// Plots a normalized histogram of actual values from the training and test datasets.
///
/// # Arguments
/// * `train_targets` - Actual values from the training dataset.
/// * `test_targets` - Actual values from the test dataset.
/// * `file_name` - Filename where the plot will be saved.
pub fn plot_normalized_histogram(
    train_targets: &Array1<f64>,
    test_targets: &Array1<f64>,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_value = *train_targets.iter().chain(test_targets.iter()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_value = *train_targets.iter().chain(test_targets.iter()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let num_bins = 20;
    let bin_width = (max_value - min_value) / num_bins as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption("Normalized Histogram of Target Values", ("sans-serif", 40).into_font())
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(min_value..max_value, 0.0..1.0)?;

    chart.configure_mesh().draw()?;

    let train_bin_counts = bin_counts(train_targets, min_value, bin_width, num_bins);
    let test_bin_counts = bin_counts(test_targets, min_value, bin_width, num_bins);

    // Drawing bars for each bin
    for (i, count) in train_bin_counts.iter().enumerate() {
        let x0 = min_value + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        let y = *count as f64 / train_targets.len() as f64; // Normalized count
        chart.draw_series(std::iter::once(Rectangle::new([(x0, 0.0), (x1, y)], RED.filled())))?;
    }

    for (i, count) in test_bin_counts.iter().enumerate() {
        let x0 = min_value + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        let y = *count as f64 / test_targets.len() as f64; // Normalized count
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, y)], 
            ShapeStyle {
                color: BLUE.mix(0.5).to_rgba(),  // Adjust color transparency
                filled: true, 
                stroke_width: 1
            }
        )))?;
    }

    Ok(())
}

fn bin_counts(data: &Array1<f64>, min_value: f64, bin_width: f64, num_bins: usize) -> Vec<usize> {
    let mut bins = vec![0; num_bins];
    for &value in data.iter() {
        let bin_index = ((value - min_value) / bin_width).floor() as usize;
        if bin_index < bins.len() {
            bins[bin_index] += 1;
        }
    }
    bins
}