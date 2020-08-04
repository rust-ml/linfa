use ndarray::{Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_rand::rand::Rng;
use num_traits::float::FloatConst;

/// Computes a similarity matrix with gaussian kernel and scaling parameter `eps`
///
/// The generated matrix is a upper triangular matrix with dimension NxN (number of observations) and contains the similarity between all permutations of observations
/// similarity
pub fn to_gaussian_similarity(
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    eps: f64,
) -> Array2<f64> {
    let n_observations = observations.len_of(Axis(0));
    let mut similarity = Array2::eye(n_observations);

    for i in 0..n_observations {
        for j in 0..n_observations {
            let a = observations.row(i);
            let b = observations.row(j);

            let distance = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powf(2.0))
                .sum::<f64>();

            similarity[(i, j)] = (-distance / eps).exp();
        }
    }

    similarity
}
///
/// Generates a three dimension swiss roll, centered at the origin with height `height` and
/// outwards speed `speed`
pub fn generate_swissroll(
    height: f64,
    speed: f64,
    n_points: usize,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let mut roll: Array2<f64> = Array2::zeros((n_points, 3));

    for i in 0..n_points {
        let z = rng.gen_range(0.0, height);
        let phi: f64 = rng.gen_range(0.0, 10.0);
        //let offset: f64 = rng.gen_range(-0.5, 0.5);
        let offset = 0.0;

        let x = speed * phi * phi.cos() + offset;
        let y = speed * phi * phi.sin() + offset;

        roll[(i, 0)] = x;
        roll[(i, 1)] = y;
        roll[(i, 2)] = z;
    }
    roll
}

pub fn generate_convoluted_rings(
    rings: &[(f64, f64)],
    n_points: usize,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let n_points = (n_points as f32 / rings.len() as f32).ceil() as usize;
    let mut array = Array2::zeros((n_points * rings.len(), 3));

    for (n, (start, end)) in rings.into_iter().enumerate() {
        // inner circle
        for i in 0..n_points {
            let r: f64 = rng.gen_range(start, end);
            let phi: f64 = rng.gen_range(0.0, f64::PI() * 2.0);
            let theta: f64 = rng.gen_range(0.0, f64::PI() * 2.0);

            let x = theta.sin() * phi.cos() * r;
            let y = theta.sin() * phi.sin() * r;
            let z = theta.cos() * r;

            array[(n * n_points + i, 0)] = x;
            array[(n * n_points + i, 1)] = y;
            array[(n * n_points + i, 2)] = z;
        }
    }

    array
}
