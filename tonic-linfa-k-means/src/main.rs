use std::path::PathBuf;
use structopt::StructOpt;
use tonic_linfa_k_means::{Store};
use tonic_linfa_k_means::server::{KMeansProto, KMeansServer};
use tonic::transport::Server;

/// This is CLI for starting linfa grpc server
#[derive(Debug, StructOpt)]
struct ServerOptions {
    #[structopt(short = "p", long = "port", default_value = "8000")]
    /// Start listening on a port, Default: 8000
    port: String,
    #[structopt(short="f", long = "load-centroids", parse(from_os_str))]
    /// Load centroids from serialized json Array2<f64>
    centroids_path: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Initialize centroids
    let opt = ServerOptions::from_args();
    let store = Store::load_json(opt.centroids_path)
        .expect("failed to load from input file");

    let (n_centroids, n_features) = store.centroids.dim();
    println!("Loaded {} centroids with {} features", n_centroids, n_features);

    // Initialize server
    let addr = format!("[::1]:{}", opt.port).parse()?;
    let kmeans = KMeansProto::new(store);
    println!("Starting server on {}", &addr);
    Server::builder()
        .add_service(KMeansServer::new(kmeans))
        .serve(addr)
        .await?;

    Ok(())
}
    
