use tonic_linfa_k_means::server::centroids::{
    client::{KMeansClient},
    Observation, Centroid,
};


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = KMeansClient::connect("http://[::1]:8000").await?;

    let request = tonic::Request::new(Observation {
        features: vec![1.0, 2.1],
    });

    let response = client.find_centroid(request).await?;

    println!("RESPONSE={:?}", response);

    Ok(())
}