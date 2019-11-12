pub mod server;

use std::path::PathBuf;
use linfa_k_means::KMeans;

pub struct Store {
    pub kmeans: KMeans,
}

impl Store {
    pub fn load(path: PathBuf) -> Result<Self, anyhow::Error> {
        Ok(Self {
            kmeans: KMeans::load(path)?,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{s, array, ArrayView2};


    #[test]
    fn load_from_file() {
        Store::load("../test/centroids.json".into()).expect("failed to load from input file");
    }


    #[test]
    fn integration() {
        let store = Store::load("../test/centroids.json".into())
            .expect("failed to load from input file");

        let sl = s![0..1, ..];
        let observations: ArrayView2<f64> = 
            store.kmeans.centroids().unwrap().slice(sl);
        assert_eq!(
            store.kmeans.predict(&observations),
            array![0]
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
