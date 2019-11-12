use super::Store;
use ndarray::Array;
use tonic::{Request, Response, Status, Code};

pub mod centroids {
    // The string specified here must match the protos package name
    tonic::include_proto!("ml");
}

use centroids::{server::ClusteringService, 
    PredictRequest, PredictResponse,
    PredictBatchRequest, PredictBatchResponse,
};

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

        let features = request.into_inner().features;
        let observation = Array::from_shape_vec((1, features.len()), features)
            .map_err(|err| Status::new(Code::InvalidArgument, err.to_string()))?;

        if let Some(cluster_index) = self.store.kmeans.predict(&observation).get(0) {
            Ok(Response::new(
                PredictResponse { cluster_index: *cluster_index as i32 }
            ))
        } else {
            Err(Status::new(Code::Internal, "KMeans::predict did not return value"))
        }
    }

    async fn predict_batch(
        &self,
        request: Request<PredictBatchRequest>, // Accept request of type HelloRequest
    ) -> Result<Response<PredictBatchResponse>, Status> {
        // Return an instance of type HelloReply
        //dbg!("Got a request: {:?}", &request);

        let observations = request.into_inner().observations;
    
        let cluster_indexes = observations.into_iter()
            .map(|obs| Array::from_shape_vec((1, obs.features.len()), obs.features).unwrap() )
            .map(|observation| *self.store.kmeans.predict(&observation).get(0).unwrap() as i32)
            .collect::<Vec<i32>>();

        Ok(Response::new(
            PredictBatchResponse { cluster_indexes  }
        ))
    }
}
