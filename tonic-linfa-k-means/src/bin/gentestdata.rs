use ndarray::array;
use serde_json;

fn main() {
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    println!("{}", serde_json::to_string(&expected_centroids).unwrap());
}
