use linfa::{
    dataset::DatasetBase,
    traits::{Fit, Predict},
};
use linfa_ica::fast_ica::{FastIca, GFunc};
use ndarray::{array, stack};
use ndarray::{Array, Array2, Axis};
use ndarray_npy::write_npy;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_isaac::Isaac64Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Create sample dataset for the model
    // `s_original` has the unmixed sources (we merely have it to save to disk)
    // `s` is the mixed source that will be unmixed using ICA
    // Shape of the data will be (2000 x 2)
    let (sources_original, sources_mixed) = create_data();

    // Fitting the model
    // We set the G function used in the approximation of neg-entropy as logcosh
    // with its alpha value as 1
    // `ncomponents` is not set, it will be automatically be assigned 2 from
    // the input
    let ica = FastIca::new().gfunc(GFunc::Logcosh(1.0));
    let ica = ica.fit(&DatasetBase::from(sources_mixed.view()))?;

    // Here we unmix the data to recover back the original signals
    let sources_ica = ica.predict(&sources_mixed);

    // Saving to disk
    write_npy("sources_original.npy", sources_original).expect("Failed to write .npy file");
    write_npy("sources_mixed.npy", sources_mixed).expect("Failed to write .npy file");
    write_npy("sources_ica.npy", sources_ica).expect("Failed to write .npy file");

    Ok(())
}

// Helper function to create two signals (sources) and mix them together
// as input for the ICA model
fn create_data() -> (Array2<f64>, Array2<f64>) {
    let nsamples = 2000;

    // Creating a sine wave signal
    let source1 = Array::linspace(0., 8., nsamples).mapv(|x| (2f64 * x).sin());

    // Creating a sawtooth signal
    let source2 = Array::linspace(0., 8., nsamples).mapv(|x| {
        let tmp = (4f64 * x).sin();
        if tmp > 0. {
            return 1.;
        }
        -1.
    });

    // Column stacking both the signals
    let mut sources_original = stack![
        Axis(1),
        source1.insert_axis(Axis(1)),
        source2.insert_axis(Axis(1))
    ];

    // Adding noise to the signals
    let mut rng = Isaac64Rng::seed_from_u64(42);
    sources_original +=
        &Array::random_using((2000, 2), Uniform::new(0.0, 1.0), &mut rng).mapv(|x| x * 0.2);

    // Mixing the two signals
    let mixing = array![[1., 1.], [0.5, 2.]];
    let sources_mixed = sources_original.dot(&mixing.t());

    (sources_original, sources_mixed)
}
