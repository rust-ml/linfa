use serde_json;
use structopt::StructOpt;
use rand::{self, Rng};
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
    pub fn new(features: Vec<f64>) -> Self {
        Self { features }
    }
}

fn main() {
    let opt = Opt::from_args();
    let mut rng = rand::thread_rng();
    let gen_sample = |_| (0..opt.features).map(|_| rng.gen_range(-100.0, 100.0)).collect::<Vec<f64>>();

    let samples = (0..opt.samples).map(gen_sample).map(Sample::new).collect::<Vec<Sample>>();
    println!("{}", serde_json::to_string(&samples).unwrap());
}
