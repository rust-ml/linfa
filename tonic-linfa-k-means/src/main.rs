use std::path::PathBuf;
use structopt::StructOpt;
use serde::{Serialize, Deserialize};
use serde_json;
use linfa_k_means::{closest_centroid};
use ndarray::{ArrayBase, Ix2, Data, Array, Array2};
use tokio::sync::Mutex;
use std::sync::Arc;

/// This is CLI for starting linfa grpc server
#[derive(Debug, StructOpt)]
struct ServerOptions {
    #[structopt(short = "p", long = "port", default_value = "8000")]
    /// Start listening on a port, Default: 8000
    port: String,
    #[structopt(short="f", long = "load-centroids", parse(from_os_str))]
    /// Load centroids from file
    centroids_path: PathBuf,
}

struct Store {
    centroids: Array2<f64>
}

struct ArcStore(Arc<Mutex<Store>>);

impl Store {
    fn load_json(path: PathBuf) -> Result<Self, anyhow::Error> {
        let file = std::fs::File::open(path)?;
        let centroids_vec: Vec<f64> = serde_json::from_reader(file)?;
        let n_centroids = centroids_vec.len();
        Ok(Self{
            centroids: Array::from_shape_vec((n_centroids, 2), centroids_vec)?
        })
    }
}

fn main()  {
    let opt = ServerOptions::from_args();
    let store = Store::load_json(opt.centroids_path)
        .expect("failed to load from input file");

//compute_centroids
}
