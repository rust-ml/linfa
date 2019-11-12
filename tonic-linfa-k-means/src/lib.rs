pub mod server;

use ndarray::Array2;
use std::path::PathBuf;

pub struct Store {
    pub centroids: Array2<f64>,
}

impl Store {
    pub fn load_json(path: PathBuf) -> Result<Self, anyhow::Error> {
        let file = std::fs::File::open(path)?;
        Ok(Self {
            centroids: serde_json::from_reader(file)?,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use linfa_k_means::closest_centroid;
    use super::server::{ClusteringServiceServer, KMeansProto};
    use tonic::transport::Server;


    #[test]
    fn load_from_file() {
        Store::load_json("./test/centroids.json".into()).expect("failed to load from input file");
    }

    #[test]
    fn integration() {
        let store = Store::load_json("./test/centroids.json".into())
            .expect("failed to load from input file");

        assert_eq!(
            closest_centroid(&store.centroids, &store.centroids.row(0)),
            0
        );
    }
/*
    #[test]
    fn tonic() {
        let store = Store::load_json("./test/centroids.json".into())
            .expect("failed to load from input file");
        
        let addr = "[::1]:50001".parse().expect("address");
        let kmeans = KMeansProto::new(store);
        let server = Server::builder()
            .add_service(ClusteringServiceServer::new(kmeans))
            .serve(addr);

        tokio::runtime::block_on(server)
    }*/

}
