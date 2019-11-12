use super::Store;
use ndarray::Array;
use tonic::{Request, Response, Status};
use linfa_k_means::closest_centroid;

pub mod centroids {
    // The string specified here must match the protos package name
    tonic::include_proto!("ml");
}

use centroids::{server::ClusteringService, PredictRequest, PredictResponse};

pub use centroids::server::ClusteringServiceServer;

pub struct KMeansProto {
    store: Store,
}

impl KMeansProto {
    pub fn new(store: Store) -> Self {
        Self {
            store
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
        //dbg!("Got a request: {:?}", &request);

        let observation = Array::from(request.into_inner().features);

        let reply = PredictResponse {
            cluster_index: 
                closest_centroid(&self.store.centroids, &observation) as i32, 
        };

        Ok(Response::new(reply)) // Send back our formatted greeting
    }
}
