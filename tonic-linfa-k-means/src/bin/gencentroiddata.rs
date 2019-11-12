use serde_json;
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    #[structopt(short = "f", long="features", default_value = "2")]
    features: usize,
    #[structopt(short = "c", long="centroids", default_value = "100")]
    centroids: usize
}


fn main() {
    let opt = Opt::from_args();
    let expected_centroids = Array::random((opt.centroids, opt.features), Uniform::new(-100.0, 100.0));
    println!("{}", serde_json::to_string(&expected_centroids).unwrap());
}
