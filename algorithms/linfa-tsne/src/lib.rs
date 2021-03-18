use ndarray::Array2;
use linfa::{dataset::DatasetBase, traits::Transformer};

pub struct TSne {
    embedding_size: usize,
    theta: f32,
    perplexity: f32,
    max_iter: usize,
}

impl TSne {
    /// Create a t-SNE param set with given embedding size
    ///
    /// # Defaults:
    ///  * `theta`: 0.5
    ///  * `perplexity`: 1.0
    ///  * `max_iter`: 2000
    pub fn embedding_size(embedding_size: usize) -> TSne {
        TSne {
            embedding_size,
            theta: 0.5,
            perplexity: 1.0,
            max_iter: 2000,
        }
    }

    /// Set the approximation value of the Barnes Hut algorith
    pub fn theta(mut self, theta: f32) -> Self {
        self.theta = theta;

        self
    }

    /// Set the perplexity of the t-SNE algorithm
    pub fn perplexity(mut self, perplexity: f32) -> Self {
        self.perplexity = perplexity;

        self
    }

    /// Set the maximal number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;

        self
    }
}

impl Transformer<Array2<f32>, Array2<f32>> for TSne
{
    fn transform(&self, mut data: Array2<f32>) -> Array2<f32> {
        let (nfeatures, nsamples) = (data.ncols(), data.nrows());

        let mut data = data.as_slice_mut().unwrap();
        let mut y = vec![0.0; nsamples * self.embedding_size];

        dbg!(&nfeatures, &nsamples);
        dbg!(&data);

        bhtsne::run(
            &mut data, 
            nsamples,
            nfeatures,
            &mut y,
            self.embedding_size,
            self.perplexity,
            self.theta,
            false,
            self.max_iter as u64,
            250,
            250,
        );

        dbg!(&y);

        Array2::from_shape_vec((nsamples, self.embedding_size), y)
            .unwrap()
    }
}

impl<T> Transformer<DatasetBase<Array2<f32>, T>, DatasetBase<Array2<f32>, T>> for TSne {
    fn transform(&self, ds: DatasetBase<Array2<f32>, T>) -> DatasetBase<Array2<f32>, T> {
        let DatasetBase {
            records,
            targets,
            ..
        } = ds;

        let new_records = self.transform(records);

        DatasetBase::new(new_records, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iris_separate() {
        let dataset = linfa_datasets::iris();
        let data = dataset.records.map(|x| *x as f32);

        let _embedded_data = TSne::embedding_size(2)
            //.perplexity(15.0)
            .theta(0.0)
            .transform(data);
    }
}
