use ndarray::{Array2, Ix2, Data, Axis, ArrayBase};

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

            let distance = a.iter().zip(b.iter()).map(|(x,y)| (x-y).powf(2.0))
                .sum::<f64>();

            similarity[(i,j)] = (-distance / eps).exp();
        }
    }

    similarity
}  
