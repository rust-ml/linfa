use serde_json;
use structopt::StructOpt;
use ndarray::{Array, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Serialize};

#[derive(StructOpt)]
struct Opt {
    #[structopt(short = "f", long="features", default_value = "2")]
    features: usize,
    #[structopt(short = "s", long="samples", default_value = "1000")]
    samples: usize
}

#[derive(Serialize)]
struct Sample {
    features: Vec<f64>
}

impl Sample {
    pub fn new(features: Array1<f64>) -> Self {
        Self { 
            features: features.to_vec()
        }
    }
}

fn main() {
    let opt = Opt::from_args();
    let distr = Uniform::new(-100.0, 100.0);
    let gen_sample = |_| Array::random(opt.features, distr);

    let samples = (0..opt.samples)
        .map(gen_sample)
        .map(Sample::new)
        .collect::<Vec<Sample>>();
    println!("{}", serde_json::to_string(&samples).unwrap());
}
