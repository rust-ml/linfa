use serde_json;
use structopt::StructOpt;
use ndarray::{Array, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Serialize};
use rand::Rng;

#[derive(StructOpt)]
struct Opt {
    #[structopt(short = "f", long="features", default_value = "2")]
    features: usize,
    #[structopt(short = "s", long="samples", default_value = "1000")]
    samples: usize,
    /// In the batch mode will generate batches number of samples of different size
    batch: bool,
    #[structopt(short = "b", long="batches", default_value = "1000")]
    batches: usize
}

#[derive(Serialize)]
struct Observation {
    features: Vec<f64>
}

impl Observation {
    pub fn new(features: Array1<f64>) -> Self {
        Self { 
            features: features.to_vec()
        }
    }
}

#[derive(Serialize)]
struct Batch {
    observations: Vec<Observation>
}

impl Batch {
    pub fn new(observations: Vec<Observation>) -> Self {
        Self { 
            observations
        }
    }
}

fn main() {
    let opt = Opt::from_args();
    let distr = Uniform::new(-100.0, 100.0);
    let gen_sample = |_| Array::random(opt.features, distr);

    let gen_samples = |n_samples| (0..n_samples)
        .map(gen_sample)
        .map(Observation::new)
        .collect::<Vec<Observation>>();
    
    if !opt.batch {
        println!("{}", serde_json::to_string(&gen_samples(opt.samples)).unwrap());
    } else {
        let mut rng = rand::thread_rng();

        let batches = 
            (0..opt.batches)
                .map(|_| rng.gen_range(1, opt.samples))
                .map(gen_samples)
                .map(Batch::new)
                .collect::<Vec<Batch>>();
        println!("{}", serde_json::to_string(&batches).unwrap());
    }
}
