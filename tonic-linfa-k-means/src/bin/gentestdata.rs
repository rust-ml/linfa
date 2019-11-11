use serde_json;
use linfa_k_means::{closest_centroid};
use ndarray::{array};


fn main() {
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    println!("{}", serde_json::to_string(&expected_centroids).unwrap());
}