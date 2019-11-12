use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use structopt::StructOpt;
use linfa_k_means::KMeans;


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
    let mut kmeans = KMeans::new(None, None);
    let observations = Array::random((opt.centroids, opt.features), Uniform::new(-100.0, 100.0));
    kmeans.fit(opt.centroids, &observations, &mut ndarray_rand::rand::thread_rng());
    kmeans.save(opt.output).unwrap();
}
