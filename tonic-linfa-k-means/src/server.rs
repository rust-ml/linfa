use super::Store;
use ndarray::Array;
use std::sync::Arc;
use tonic::{Request, Response, Status};

pub mod centroids {
    // The string specified here must match the protos package name
    tonic::include_proto!("ml");
}

use centroids::{server::ClusteringService, PredictRequest, PredictResponse};

pub use centroids::server::ClusteringServiceServer;

pub struct KMeansProto {
    store: Arc<Store>,
}

impl KMeansProto {
    pub fn new(store: Store) -> Self {
        Self {
            store: Arc::new(store),
        }
    }
}

#[tonic::async_trait]
impl ClusteringService for KMeansProto {
    async fn predict(
        &self,
        request: Request<PredictRequest>, // Accept request of type HelloRequest
    ) -> Result<Response<PredictResponse>, Status> {
        // Return an instance of type HelloReply
        println!("Got a request: {:?}", request);

        let observation = Array::from(request.into_inner().features);

        // closest_centroid(&self.store.clone().centroids)
        let reply = PredictResponse {
            cluster_index: 1, // We must use .into_inner() as the fields of gRPC requests and responses are private
        };

        Ok(Response::new(reply)) // Send back our formatted greeting
    }
}
