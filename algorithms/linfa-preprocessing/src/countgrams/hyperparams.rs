use crate::PreprocessingError;
use linfa::ParamGuard;
use regex::Regex;
use std::cell::{Ref, RefCell};
use std::collections::HashSet;

/// Count vectorizer: learns a vocabulary from a sequence of documents (or file paths) and maps each
/// vocabulary entry to an integer value, producing a [CountVectorizer](crate::CountVectorizer) that can
/// be used to count the occurrences of each vocabulary entry in any sequence of documents. Alternatively a user-specified vocabulary can
/// be used for fitting.
///
/// ### Attributes
///
/// If a user-defined vocabulary is used for fitting then the following attributes will not be considered during the fitting phase but
/// they will still be used by the [CountVectorizer](crate::CountVectorizer) to transform any text to be examined.
///
/// * `split_regex`: the regex espression used to split decuments into tokens. Defaults to r"\\b\\w\\w+\\b", which selects "words", using whitespaces and
/// punctuation symbols as separators.
/// * `convert_to_lowercase`: if true, all documents used for fitting will be converted to lowercase. Defaults to `true`.
/// * `n_gram_range`: if set to `(1,1)` single tokens will be candidate vocabulary entries, if `(2,2)` then adjacent token pairs will be considered,
///    if `(1,2)` then both single tokens and adjacent token pairs will be considered, and so on. The definition of token depends on the
///    regex used fpr splitting the documents. The default value is `(1,1)`.
/// * `normalize`: if true, all charachters in the documents used for fitting will be normalized according to unicode's NFKD normalization. Defaults to `true`.
/// * `document_frequency`: specifies the minimum and maximum (relative) document frequencies that each vocabulary entry must satisfy. Defaults to `(0., 1.)` (i.e. 0% minimum and 100% maximum)
/// * `stopwords`: optional list of entries to be excluded from the generated vocabulary. Defaults to `None`
#[derive(Clone, Debug)]
pub struct CountVectorizerValidParams {
    convert_to_lowercase: bool,
    split_regex_expr: String,
    split_regex: RefCell<Option<Regex>>,
    n_gram_range: (usize, usize),
    normalize: bool,
    document_frequency: (f32, f32),
    stopwords: Option<HashSet<String>>,
}

impl CountVectorizerValidParams {
    pub fn convert_to_lowercase(&self) -> bool {
        self.convert_to_lowercase
    }

    pub fn split_regex(&self) -> Ref<'_, Regex> {
        Ref::map(self.split_regex.borrow(), |x| x.as_ref().unwrap())
    }

    pub fn n_gram_range(&self) -> (usize, usize) {
        self.n_gram_range
    }

    pub fn normalize(&self) -> bool {
        self.normalize
    }

    pub fn document_frequency(&self) -> (f32, f32) {
        self.document_frequency
    }

    pub fn stopwords(&self) -> &Option<HashSet<String>> {
        &self.stopwords
    }
}

#[derive(Clone, Debug)]
pub struct CountVectorizerParams(CountVectorizerValidParams);

impl std::default::Default for CountVectorizerParams {
    fn default() -> Self {
        Self(CountVectorizerValidParams {
            convert_to_lowercase: true,
            split_regex_expr: r"\b\w\w+\b".to_string(),
            split_regex: RefCell::new(None),
            n_gram_range: (1, 1),
            normalize: true,
            document_frequency: (0., 1.),
            stopwords: None,
        })
    }
}

impl CountVectorizerParams {
    ///If true, all documents used for fitting will be converted to lowercase.
    pub fn convert_to_lowercase(mut self, convert_to_lowercase: bool) -> Self {
        self.0.convert_to_lowercase = convert_to_lowercase;
        self
    }

    /// Sets the regex espression used to split decuments into tokens
    pub fn split_regex(mut self, regex_str: &str) -> Self {
        self.0.split_regex_expr = regex_str.to_string();
        self
    }

    /// If set to `(1,1)` single tokens will be candidate vocabulary entries, if `(2,2)` then adjacent token pairs will be considered,
    /// if `(1,2)` then both single tokens and adjacent token pairs will be considered, and so on. The definition of token depends on the
    /// regex used fpr splitting the documents.
    ///
    /// `min_n` should not be greater than `max_n`
    pub fn n_gram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.0.n_gram_range = (min_n, max_n);
        self
    }

    /// If true, all charachters in the documents used for fitting will be normalized according to unicode's NFKD normalization.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.0.normalize = normalize;
        self
    }

    /// Specifies the minimum and maximum (relative) document frequencies that each vocabulary entry must satisfy.
    /// `min_freq` and `max_freq` must lie in `0..=1` and `min_freq` should not be greater than `max_freq`
    pub fn document_frequency(mut self, min_freq: f32, max_freq: f32) -> Self {
        self.0.document_frequency = (min_freq, max_freq);
        self
    }

    /// List of entries to be excluded from the generated vocabulary.
    pub fn stopwords<T: ToString>(mut self, stopwords: &[T]) -> Self {
        self.0.stopwords = Some(stopwords.iter().map(|t| t.to_string()).collect());
        self
    }
}

impl ParamGuard for CountVectorizerParams {
    type Checked = CountVectorizerValidParams;
    type Error = PreprocessingError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        let (n_gram_min, n_gram_max) = self.0.n_gram_range;
        let (min_freq, max_freq) = self.0.document_frequency;

        if n_gram_min == 0 || n_gram_max == 0 {
            Err(PreprocessingError::InvalidNGramBoundaries(
                n_gram_min, n_gram_max,
            ))
        } else if n_gram_min > n_gram_max {
            Err(PreprocessingError::FlippedNGramBoundaries(
                n_gram_min, n_gram_max,
            ))
        } else if min_freq < 0. || max_freq < 0. {
            Err(PreprocessingError::InvalidDocumentFrequencies(
                min_freq, max_freq,
            ))
        } else if max_freq < min_freq {
            Err(PreprocessingError::FlippedDocumentFrequencies(
                min_freq, max_freq,
            ))
        } else {
            *self.0.split_regex.borrow_mut() = Some(Regex::new(&self.0.split_regex_expr)?);

            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
