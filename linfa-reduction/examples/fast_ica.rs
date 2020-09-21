use linfa_reduction::fast_ica::{FastIca, GFunc};
use ndarray::{array, stack};
use ndarray::{Array, Array2, Axis};
use ndarray_npy::write_npy;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_isaac::Isaac64Rng;

fn main() {
    let (s_original, s) = create_data();

    let ica = FastIca::new().set_gfunc(GFunc::Logcosh(1.0));
    let ica = ica.fit(&s).unwrap();
    let output = ica.transform(&s);

    write_npy("s_original.npy", s_original).expect("Failed to write .npy file");
    write_npy("s_mixed.npy", s).expect("Failed to write .npy file");
    write_npy("s_output.npy", output).expect("Failed to write .npy file");
}

fn create_data() -> (Array2<f64>, Array2<f64>) {
    let n_samples = 2000;

    let s1 = Array::linspace(0., 8., n_samples).mapv(|x| (2f64 * x).sin());
    let s2 = Array::linspace(0., 8., n_samples).mapv(|x| {
        let tmp = (4f64 * x).sin();
        if tmp > 0. {
            return 1.;
        }
        -1.
    });
    let mut s_original = stack![Axis(1), s1.insert_axis(Axis(1)), s2.insert_axis(Axis(1))];
    let mut rng = Isaac64Rng::seed_from_u64(42);
    s_original +=
        &Array::random_using((2000, 2), Uniform::new(0.0, 1.0), &mut rng).mapv(|x| x * 0.2);
    let mixing = array![[1., 1.], [0.5, 2.]];
    let s = s_original.dot(&mixing.t());

    (s_original, s)
}
