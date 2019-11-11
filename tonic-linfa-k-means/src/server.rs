use tonic::{Request, Response, Status};
use super::Store;
use std::sync::Arc;
use linfa_k_means::{closest_centroid};
use ndarray::Array;


pub mod centroids {
  tonic::include_proto!("centroids"); // The string specified here must match the proto package name
}

use centroids::{
    server::{KMeans},
    Observation, Centroid,
};

pub use centroids::server::KMeansServer;

pub struct KMeansProto {
  store: Arc<Store>
}

impl KMeansProto {
  pub fn new(store: Store) -> Self {
    Self {
      store: Arc::new(store)
    }
  }
}

#[tonic::async_trait]
impl KMeans for KMeansProto {
    async fn find_centroid(
        &self,
        request: Request<Observation>, // Accept request of type HelloRequest
    ) -> Result<Response<Centroid>, Status> { // Return an instance of type HelloReply
        println!("Got a request: {:?}", request);

        let observation = Array::from_vec(request.into_inner().features);

        // closest_centroid(&self.store.clone().centroids)
        let reply = Centroid {
            cluster_index: 1, // We must use .into_inner() as the fields of gRPC requests and responses are private
        };

        Ok(Response::new(reply)) // Send back our formatted greeting
    }
}

