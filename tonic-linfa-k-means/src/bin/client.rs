use tonic_linfa_k_means::server::centroids::{client::ClusteringServiceClient, PredictRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = ClusteringServiceClient::connect("http://[::1]:8000").await?;

    let request = tonic::Request::new(PredictRequest {
        features: vec![1.0, 2.1],
    });

    let response = client.predict(request).await?;

    println!("RESPONSE={:?}", response);

    Ok(())
}
