use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use structopt::StructOpt;
use linfa_clustering::{KMeans, KMeansHyperParams};


#[derive(StructOpt)]
struct Opt {
    #[structopt(short = "f", long="features", default_value = "2")]
    features: usize,
    #[structopt(short = "c", long="centroids", default_value = "100")]
    centroids: usize,
    #[structopt(short = "o", long="output", default_value = "./centroids.json")]
    output: std::path::PathBuf
}


fn main() {
    let opt = Opt::from_args();
    // We just need a model instance, how we trained it doesn't have any influence
    // on inference performance
    let hyperparams = KMeansHyperParams::new(opt.centroids).build();
    let observations = Array::random((opt.centroids, opt.features), Uniform::new(-100.0, 100.0));
    let kmeans = KMeans::fit(hyperparams, &observations, &mut ndarray_rand::rand::thread_rng());

    let writer = std::fs::File::create(opt.output).unwrap();
    serde_json::to_writer(writer, &kmeans).unwrap();
}
