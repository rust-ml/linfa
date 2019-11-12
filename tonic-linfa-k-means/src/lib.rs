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

    #[test]
    fn load_from_file() {
        Store::load_json("./test/centroids.json".into()).expect("failed to load from input file");
    }

    fn integration() {
        let store = Store::load_json("./test/centroids.json".into())
            .expect("failed to load from input file");

        assert_eq!(
            closest_centroid(&store.centroids, &store.centroids.row(0)),
            0
        );
    }
}
