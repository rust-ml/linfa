use ndarray::Array2;
use linfa::{Float, dataset::DatasetBase, traits::Transformer};

pub struct TSne<F: Float> {
    embedding_size: usize,
    theta: F,
    perplexity: F,
    max_iter: usize,
}

impl<F: Float> TSne<F> {
    /// Create a t-SNE param set with given embedding size
    ///
    /// # Defaults:
    ///  * `theta`: 0.5
    ///  * `perplexity`: 1.0
    ///  * `max_iter`: 2000
    pub fn embedding_size(embedding_size: usize) -> TSne<F> {
        TSne {
            embedding_size,
            theta: F::from(0.5).unwrap(),
            perplexity: F::from(5.0).unwrap(),
            max_iter: 2000,
        }
    }

    /// Set the approximation value of the Barnes Hut algorith
    pub fn theta(mut self, theta: F) -> Self {
        self.theta = theta;

        self
    }

    /// Set the perplexity of the t-SNE algorithm
    pub fn perplexity(mut self, perplexity: F) -> Self {
        self.perplexity = perplexity;

        self
    }

    /// Set the maximal number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;

        self
    }
}

impl<F: Float> Transformer<Array2<F>, Array2<F>> for TSne<F>
{
    fn transform(&self, mut data: Array2<F>) -> Array2<F> {
        let (nfeatures, nsamples) = (data.ncols(), data.nrows());

        let mut data = data.as_slice_mut().unwrap();
        let mut y = vec![F::zero(); nsamples * self.embedding_size];

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

impl<T, F: Float> Transformer<DatasetBase<Array2<F>, T>, DatasetBase<Array2<F>, T>> for TSne<F> {
    fn transform(&self, ds: DatasetBase<Array2<F>, T>) -> DatasetBase<Array2<F>, T> {
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
    use std::io::Write;
    use ndarray::Axis;
    use linfa_reduction::Pca;
    use linfa::traits::{Fit, Predict};

    #[test]
    fn iris_separate() {
        let ds = linfa_datasets::winequality();
        let records = Pca::params(8).whiten(true).fit(&ds).predict(&ds);

        let DatasetBase { targets, .. } = ds;
        let records = records.map(|x| *x as f32);
        let targets = targets.map(|x| if *x > 6 { 2.0 } else { 1.0 });

        let records = TSne::embedding_size(2)
            .perplexity(20.0)
            .theta(0.5)
            .transform(records);

        dbg!(&targets);
        let mut f = std::fs::File::create("iris.dat").unwrap();

        for (x, y) in records.axis_iter(Axis(0)).zip(targets.iter()) {
            f.write(format!("{} {} {}\n", x[0], x[1], y).as_bytes()).unwrap();
        }
    }
}
