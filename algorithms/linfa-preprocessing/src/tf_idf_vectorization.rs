//! Count vectorization methods

use crate::count_vectorization::{CountVectorizer, FittedCountVectorizer};
use crate::error::Result;
use ndarray::{Array2, ArrayBase, Data, Ix1, Zip};

#[derive(Clone)]
/// Methods for computing the inverse document frequency of a vocabulary entry
pub enum TfIdfMethod {
    /// Computes the idf as `log(1+n/1+document_frequency) + 1`. The "plus ones" inside the log
    /// add an artificial document containing every vocabulary entry, preventing divisions by zero.
    /// The "plus one" after the log allows vocabulary entries that appear in every document to still be considered with
    /// a weight of one instead of being completely discarded.
    Smooth,
    /// Computes the idf as `log(n/document_frequency) +1`. The "plus one" after the log allows vocabulary entries that appear in every document to still be considered with
    /// a weight of one instead of being completely discarded. If a vocabulary entry has zero document frequency this will produce a division by zero.
    NonSmooth,
    /// Textbook definition of idf, computed as `log(n/ 1 + document_frequency)` which prevents divisions by zero and discards entries that appear in every document.
    Textbook,
}

impl TfIdfMethod {
    pub fn compute_idf(&self, n: usize, df: usize) -> f64 {
        match self {
            TfIdfMethod::Smooth => (1. + n as f64) / (1. + df as f64).ln() + 1.,
            TfIdfMethod::NonSmooth => (n as f64 / df as f64).ln() + 1.,
            TfIdfMethod::Textbook => (n as f64 / (1. + df as f64)).ln(),
        }
    }
}

pub struct TfIdfVectorizer {
    count_vectorizer: CountVectorizer,
    method: TfIdfMethod,
}

impl std::default::Default for TfIdfVectorizer {
    fn default() -> Self {
        Self {
            count_vectorizer: CountVectorizer::default(),
            method: TfIdfMethod::Textbook,
        }
    }
}

impl TfIdfVectorizer {
    pub fn remove_punctuation(self, remove_punctuation: bool) -> Self {
        Self {
            count_vectorizer: self.count_vectorizer.remove_punctuation(remove_punctuation),
            method: self.method,
        }
    }

    pub fn convert_to_lowercase(self, convert_to_lowercase: bool) -> Self {
        Self {
            count_vectorizer: self
                .count_vectorizer
                .convert_to_lowercase(convert_to_lowercase),
            method: self.method,
        }
    }

    pub fn punctuation_symbols(self, punctuation_symbols: &[char]) -> Self {
        Self {
            count_vectorizer: self
                .count_vectorizer
                .punctuation_symbols(punctuation_symbols),
            method: self.method,
        }
    }

    pub fn n_gram_range(self, min_n: usize, max_n: usize) -> Self {
        Self {
            count_vectorizer: self.count_vectorizer.n_gram_range(min_n, max_n),
            method: self.method,
        }
    }

    pub fn normalize(self, normalize: bool) -> Self {
        Self {
            count_vectorizer: self.count_vectorizer.normalize(normalize),
            method: self.method,
        }
    }

    pub fn document_frequency(self, min_freq: f32, max_freq: f32) -> Self {
        Self {
            count_vectorizer: self.count_vectorizer.document_frequency(min_freq, max_freq),
            method: self.method,
        }
    }

    /// Learns a vocabulary from the texts in `x`, according to the specified attributes and maps each
    /// vocabulary entry to an integer value, producing a [FittedTfIdfVectorizer](struct.FittedTfIdfVectorizer.html).
    ///
    /// Returns an error if:
    /// * one of the `n_gram` boundaries is set to zero or the minimum value is greater than the maximum value
    /// * if the minimum document frequency is greater than one or than the maximum frequency, or if the maximum frequecy is  
    ///   smaller than zero
    pub fn fit<T: ToString + Clone, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Result<FittedTfIdfVectorizer> {
        let fitted_vectorizer = self.count_vectorizer.fit(x)?;
        Ok(FittedTfIdfVectorizer {
            fitted_vectorizer,
            method: self.method.clone(),
        })
    }

    /// Produces a [FittedTfIdfVectorizer](struct.FittedTfIdfVectorizer.html) with the input vocabulary.
    /// All struct attributes are ignored in the fitting but will be used by the [FittedTfIdfVectorizer](struct.FittedTfIdfVectorizer.html)
    /// to transform any text to be examined. As such this will return an error in the same cases as the `fit` method.
    pub fn fit_vocabulary<T: ToString>(&self, words: &[T]) -> Result<FittedTfIdfVectorizer> {
        let fitted_vectorizer = self.count_vectorizer.fit_vocabulary(words)?;
        Ok(FittedTfIdfVectorizer {
            fitted_vectorizer,
            method: self.method.clone(),
        })
    }
}

pub struct FittedTfIdfVectorizer {
    fitted_vectorizer: FittedCountVectorizer,
    method: TfIdfMethod,
}

impl FittedTfIdfVectorizer {
    /// Constains all vocabulary entries, in the same order used by the `transform` method.
    pub fn vocabulary(&self) -> &Vec<String> {
        self.fitted_vectorizer.vocabulary()
    }

    pub fn method(&self) -> &TfIdfMethod {
        &self.method
    }

    pub fn transform<T: ToString, D: Data<Elem = T>>(&self, x: &ArrayBase<D, Ix1>) -> Array2<f64> {
        let (term_freqs, doc_freqs) = self.fitted_vectorizer.get_term_and_document_frequencies(x);
        let mut term_freqs = term_freqs.mapv(|x| x as f64);
        let inv_doc_freqs =
            doc_freqs.mapv(|doc_freq| ((1. + x.len() as f64) / (1. + doc_freq as f64)).ln() + 1.);
        for row in term_freqs.genrows_mut() {
            Zip::from(row)
                .and(&inv_doc_freqs)
                .apply(|el, inv_doc_f| *el *= *inv_doc_f);
        }
        term_freqs
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::column_for_word;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    macro_rules! assert_tf_idfs_for_word {

        ($voc:expr, $transf:expr, $(($word:expr, $counts:expr)),*) => {
            $ (
                assert_abs_diff_eq!(column_for_word!($voc, $transf, $word), $counts, epsilon=1e-3);
            )*
        }
    }

    #[test]
    fn test_tf_idf() {
        let texts = array![
            "one and two and three",
            "three and four and five",
            "seven and eight",
            "maybe ten and eleven",
            "avoid singletons: one two four five seven eight ten eleven and an and"
        ];
        let vectorizer = TfIdfVectorizer::default().fit(&texts).unwrap();
        let vocabulary = vectorizer.vocabulary();
        let transformed = vectorizer.transform(&texts);
        assert_eq!(transformed.dim(), (texts.len(), vocabulary.len()));
        assert_tf_idfs_for_word!(
            vocabulary,
            transformed,
            ("one", array![1.693, 0.0, 0.0, 0.0, 1.693]),
            ("two", array![1.693, 0.0, 0.0, 0.0, 1.693]),
            ("three", array![1.693, 1.693, 0.0, 0.0, 0.0]),
            ("four", array![0.0, 1.693, 0.0, 0.0, 1.693]),
            ("and", array![2.0, 2.0, 1.0, 1.0, 2.0]),
            ("five", array![0.0, 1.693, 0.0, 0.0, 1.693]),
            ("seven", array![0.0, 0.0, 1.693, 0.0, 1.693]),
            ("eight", array![0.0, 0.0, 1.693, 0.0, 1.693]),
            ("ten", array![0.0, 0.0, 0.0, 1.693, 1.693]),
            ("eleven", array![0.0, 0.0, 0.0, 1.693, 1.693]),
            ("an", array![0.0, 0.0, 0.0, 0.0, 2.098]),
            ("avoid", array![0.0, 0.0, 0.0, 0.0, 2.098]),
            ("singletons", array![0.0, 0.0, 0.0, 0.0, 2.098]),
            ("maybe", array![0.0, 0.0, 0.0, 2.098, 0.0])
        );
    }
}
