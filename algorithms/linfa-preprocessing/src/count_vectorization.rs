//! Count vectorization methods

use crate::error::{Error, Result};
use crate::helpers::NGramList;
use encoding::types::EncodingRef;
use encoding::DecoderTrap;
use ndarray::{Array1, ArrayBase, ArrayViewMut1, Data, Ix1};
use regex::Regex;
use sprs::{CsMat, CsVec};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::iter::IntoIterator;
use unicode_normalization::UnicodeNormalization;

#[derive(Clone)]
/// Struct that holds all vectorizer options so that they can be passed to the fitted vectorizer
pub(crate) struct VectorizerProperties {
    convert_to_lowercase: bool,
    split_regex: String,
    n_gram_range: (usize, usize),
    normalize: bool,
    document_frequency: (f32, f32),
    stopwords: Option<HashSet<String>>,
}

/// Count vectorizer: learns a vocabulary from a sequence of documents (or file paths) and maps each
/// vocabulary entry to an integer value, producing a [FittedCountVectorizer](struct.FittedCountVectorizer.html) that can
/// be used to count the occurrences of each vocabulary entry in any sequence of documents. Alternatively a user-specified vocabulary can
/// be used for fitting.
///
/// ### Attributes
///
/// If a user-defined vocabulary is used for fitting then the following attributes will not be considered during the fitting phase but
/// they will still be used by the [FittedCountVectorizer](struct.FittedCountVectorizer.html) to transform any text to be examined.
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
///
pub struct CountVectorizer {
    properties: VectorizerProperties,
}

impl std::default::Default for CountVectorizer {
    fn default() -> Self {
        Self {
            properties: VectorizerProperties {
                convert_to_lowercase: true,
                split_regex: r"\b\w\w+\b".to_string(),
                n_gram_range: (1, 1),
                normalize: true,
                document_frequency: (0., 1.),
                stopwords: None,
            },
        }
    }
}

impl CountVectorizer {
    ///If true, all documents used for fitting will be converted to lowercase.
    pub fn convert_to_lowercase(mut self, convert_to_lowercase: bool) -> Self {
        self.properties.convert_to_lowercase = convert_to_lowercase;
        self
    }

    /// Sets the regex espression used to split decuments into tokens
    pub fn split_regex(mut self, regex_str: &str) -> Self {
        self.properties.split_regex = regex_str.to_string();
        self
    }

    /// If set to `(1,1)` single tokens will be candidate vocabulary entries, if `(2,2)` then adjacent token pairs will be considered,
    /// if `(1,2)` then both single tokens and adjacent token pairs will be considered, and so on. The definition of token depends on the
    /// regex used fpr splitting the documents.
    ///
    /// `min_n` should not be greater than `max_n`
    pub fn n_gram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.properties.n_gram_range = (min_n, max_n);
        self
    }

    /// If true, all charachters in the documents used for fitting will be normalized according to unicode's NFKD normalization.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.properties.normalize = normalize;
        self
    }

    /// Specifies the minimum and maximum (relative) document frequencies that each vocabulary entry must satisfy.
    /// `min_freq` and `max_freq` must lie in [0;1] and `min_freq` should not be greater than `max_freq`
    pub fn document_frequency(mut self, min_freq: f32, max_freq: f32) -> Self {
        self.properties.document_frequency = (min_freq, max_freq);
        self
    }

    /// List of entries to be excluded from the generated vocabulary.
    pub fn stopwords<T: ToString>(mut self, stopwords: &[T]) -> Self {
        self.properties.stopwords = Some(stopwords.iter().map(|t| t.to_string()).collect());
        self
    }

    /// Learns a vocabulary from the documents in `x`, according to the specified attributes and maps each
    /// vocabulary entry to an integer value, producing a [FittedCountVectorizer](struct.FittedCountVectorizer.html).
    ///
    /// Returns an error if:
    /// * one of the `n_gram` boundaries is set to zero or the minimum value is greater than the maximum value
    /// * if the minimum document frequency is greater than one or than the maximum frequency, or if the maximum frequency is  
    ///   smaller than zero
    /// * if the regex expression for the split is invalid
    pub fn fit<T: ToString + Clone, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Result<FittedCountVectorizer> {
        let regex = validate_properties(&self.properties)?;

        // word, (integer mapping for word, document frequency for word)
        let mut vocabulary: HashMap<String, (usize, usize)> = HashMap::new();
        for string in x
            .iter()
            .map(|s| transform_string(s.to_string(), &self.properties))
        {
            self.read_document_into_vocabulary(string, &regex, &mut vocabulary);
        }

        let mut vocabulary = self.filter_vocabulary(vocabulary, x.len());
        let vec_vocabulary = hashmap_to_vocabulary(&mut vocabulary);

        Ok(FittedCountVectorizer {
            vocabulary,
            vec_vocabulary,
            properties: self.properties.clone(),
        })
    }

    /// Learns a vocabulary from the documents contained in the files in `input`, according to the specified attributes and maps each
    /// vocabulary entry to an integer value, producing a [FittedCountVectorizer](struct.FittedCountVectorizer.html).
    ///
    /// The files will be read using the specified `encoding`, and any sequence unrecognized by the encoding will be handled
    /// according to `trap`.
    ///
    /// Returns an error if:
    /// * one of the `n_gram` boundaries is set to zero or the minimum value is greater than the maximum value
    /// * if the minimum document frequency is greater than one or than the maximum frequency, or if the maximum frequency is  
    ///   smaller than zero
    /// * if the regex expression for the split is invalid
    /// * if one of the files couldn't be opened
    /// * if the trap is strict and an unrecognized sequence is encountered in one of the files
    pub fn fit_files<P: AsRef<std::path::Path>>(
        &self,
        input: &[P],
        encoding: EncodingRef,
        trap: DecoderTrap,
    ) -> Result<FittedCountVectorizer> {
        let regex = validate_properties(&self.properties)?;
        // word, (integer mapping for word, document frequency for word)
        let mut vocabulary: HashMap<String, (usize, usize)> = HashMap::new();
        let documents_count = input.len();
        for path in input {
            let mut file = std::fs::File::open(&path)?;
            let mut document_bytes = Vec::new();
            file.read_to_end(&mut document_bytes)?;
            let document = encoding::decode(&document_bytes, trap, encoding).0;
            // encoding error contains a cow string, can't just use ?, must go through the unwrap
            if document.is_err() {
                return Err(crate::error::Error::EncodingError(document.err().unwrap()));
            }
            // safe unwrap now that error has been handled
            let document = transform_string(document.unwrap(), &self.properties);
            self.read_document_into_vocabulary(document, &regex, &mut vocabulary);
        }

        let mut vocabulary = self.filter_vocabulary(vocabulary, documents_count);
        let vec_vocabulary = hashmap_to_vocabulary(&mut vocabulary);

        Ok(FittedCountVectorizer {
            vocabulary,
            vec_vocabulary,
            properties: self.properties.clone(),
        })
    }

    /// Produces a [FittedCountVectorizer](struct.FittedCountVectorizer.html) with the input vocabulary.
    /// All struct attributes are ignored in the fitting but will be used by the [FittedCountVectorizer](struct.FittedCountVectorizer.html)
    /// to transform any text to be examined. As such this will return an error in the same cases as the `fit` method.
    pub fn fit_vocabulary<T: ToString>(&self, words: &[T]) -> Result<FittedCountVectorizer> {
        validate_properties(&self.properties)?;
        let mut vocabulary: HashMap<String, (usize, usize)> = HashMap::with_capacity(words.len());
        for item in words.iter().map(|w| w.to_string()) {
            let len = vocabulary.len();
            // do not care about frequencies/stopwords if a vocabulary is given. Always 1 frequency
            vocabulary.entry(item).or_insert((len, 1));
        }
        let vec_vocabulary = hashmap_to_vocabulary(&mut vocabulary);
        Ok(FittedCountVectorizer {
            vocabulary,
            vec_vocabulary,
            properties: self.properties.clone(),
        })
    }

    /// Removes vocabulary items that do not satisfy the document frequencies constraints or if they appear in the
    /// optional stopwords test.
    /// The total number of documents is needed to convert from relative document frequencies to
    /// their absolute counterparts.
    fn filter_vocabulary(
        &self,
        vocabulary: HashMap<String, (usize, usize)>,
        n_documents: usize,
    ) -> HashMap<String, (usize, usize)> {
        let (min_df, max_df) = self.properties.document_frequency;
        let len_f32 = n_documents as f32;
        let (min_abs_df, max_abs_df) = ((min_df * len_f32) as usize, (max_df * len_f32) as usize);

        if min_abs_df == 0 && max_abs_df == n_documents {
            match &self.properties.stopwords {
                None => vocabulary,
                Some(stopwords) => vocabulary
                    .into_iter()
                    .filter(|(entry, (_, _))| !stopwords.contains(entry))
                    .collect(),
            }
        } else {
            match &self.properties.stopwords {
                None => vocabulary
                    .into_iter()
                    .filter(|(_, (_, abs_count))| {
                        *abs_count >= min_abs_df && *abs_count <= max_abs_df
                    })
                    .collect(),
                Some(stopwords) => vocabulary
                    .into_iter()
                    .filter(|(entry, (_, abs_count))| {
                        *abs_count >= min_abs_df
                            && *abs_count <= max_abs_df
                            && !stopwords.contains(entry)
                    })
                    .collect(),
            }
        }
    }

    /// Inserts all vocabulary entries learned from a single document (`doc`) into the
    /// shared `vocabulary`, setting the document frequency to one for new entries and
    /// incrementing it by one for entries which were already present.
    fn read_document_into_vocabulary(
        &self,
        doc: String,
        regex: &Regex,
        vocabulary: &mut HashMap<String, (usize, usize)>,
    ) {
        let words = regex.find_iter(&doc).map(|mat| mat.as_str()).collect();
        let list = NGramList::new(words, self.properties.n_gram_range);
        let document_vocabulary: HashSet<String> = list.into_iter().flatten().collect();
        for word in document_vocabulary {
            let len = vocabulary.len();
            // If vocabulary item was already present then increase its document frequency
            if let Some((_, freq)) = vocabulary.get_mut(&word) {
                *freq += 1;
            // otherwise set it to one
            } else {
                vocabulary.insert(word, (len, 1));
            }
        }
    }
}

/// Counts the occurrences of each vocabulary entry, learned during fitting, in a sequence of documents. Each vocabulary entry is mapped
/// to an integer value that is used to index the count in the result.
pub struct FittedCountVectorizer {
    pub(crate) vocabulary: HashMap<String, (usize, usize)>,
    pub(crate) vec_vocabulary: Vec<String>,
    pub(crate) properties: VectorizerProperties,
}

impl FittedCountVectorizer {
    /// Number of vocabulary entries learned during fitting
    pub fn nentries(&self) -> usize {
        self.vocabulary.len()
    }

    /// Given a sequence of `n` documents, produces a sparse array of size `(n, vocabulary_entries)` where column `j` of row `i`
    /// is the number of occurrences of vocabulary entry `j` in the document of index `i`. Vocabulary entry `j` is the string
    /// at the `j`-th position in the vocabulary. If a vocabulary entry was not encountered in a document, then the relative
    /// cell in the sparse matrix will be set to `None`.
    pub fn transform<T: ToString, D: Data<Elem = T>>(&self, x: &ArrayBase<D, Ix1>) -> CsMat<usize> {
        let (vectorized, _) = self.get_term_and_document_frequencies(x);
        vectorized
    }

    /// Given a sequence of `n` file names, produces a sparse array of size `(n, vocabulary_entries)` where column `j` of row `i`
    /// is the number of occurrences of vocabulary entry `j` in the document contained in the file of index `i`. Vocabulary entry `j` is the string
    /// at the `j`-th position in the vocabulary. If a vocabulary entry was not encountered in a document, then the relative
    /// cell in the sparse matrix will be set to `None`.
    ///
    /// The files will be read using the specified `encoding`, and any sequence unrecognized by the encoding will be handled
    /// according to `trap`.
    pub fn transform_files<P: AsRef<std::path::Path>>(
        &self,
        input: &[P],
        encoding: EncodingRef,
        trap: DecoderTrap,
    ) -> CsMat<usize> {
        let (vectorized, _) = self.get_term_and_document_frequencies_files(input, encoding, trap);
        vectorized
    }

    /// Contains all vocabulary entries, in the same order used by the `transform` methods.
    pub fn vocabulary(&self) -> &Vec<String> {
        &self.vec_vocabulary
    }

    /// Counts the occurrence of each vocabulary entry in each document and keeps track of the overall
    /// document frequency of each entry.
    pub(crate) fn get_term_and_document_frequencies<T: ToString, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> (CsMat<usize>, Array1<usize>) {
        let mut document_frequencies = Array1::zeros(self.vocabulary.len());
        let mut sprs_vectorized = CsMat::empty(sprs::CompressedStorage::CSR, self.vocabulary.len());
        sprs_vectorized.reserve_outer_dim_exact(x.len());
        let regex = Regex::new(&self.properties.split_regex).unwrap();
        for (_string_index, string) in x.into_iter().map(|s| s.to_string()).enumerate() {
            let row = self.analyze_document(string, &regex, document_frequencies.view_mut());
            sprs_vectorized = sprs_vectorized.append_outer_csvec(row.view());
        }
        (sprs_vectorized, document_frequencies)
    }

    /// Counts the occurrence of each vocabulary entry in each document and keeps track of the overall
    /// document frequency of each entry.
    pub(crate) fn get_term_and_document_frequencies_files<P: AsRef<std::path::Path>>(
        &self,
        input: &[P],
        encoding: EncodingRef,
        trap: DecoderTrap,
    ) -> (CsMat<usize>, Array1<usize>) {
        let mut document_frequencies = Array1::zeros(self.vocabulary.len());
        let mut sprs_vectorized = CsMat::empty(sprs::CompressedStorage::CSR, self.vocabulary.len());
        sprs_vectorized.reserve_outer_dim_exact(input.len());
        let regex = Regex::new(&self.properties.split_regex).unwrap();
        for (_file_index, file_path) in input.iter().enumerate() {
            let mut file = std::fs::File::open(&file_path).unwrap();
            let mut document_bytes = Vec::new();
            file.read_to_end(&mut document_bytes).unwrap();
            let document = encoding::decode(&document_bytes, trap, encoding).0.unwrap();
            sprs_vectorized = sprs_vectorized.append_outer_csvec(
                self.analyze_document(document, &regex, document_frequencies.view_mut())
                    .view(),
            );
        }
        (sprs_vectorized, document_frequencies)
    }

    /// Produces a sparse array which counts the occurrences of each vocbulary entry in the given document. Also increases
    /// the document frequency of all entries found.
    fn analyze_document(
        &self,
        document: String,
        regex: &Regex,
        mut doc_freqs: ArrayViewMut1<usize>,
    ) -> CsVec<usize> {
        // A dense array is needed to parse each document, since sparse arrays can be mutated only
        // if all insertions are made with increasing index. Since  vocabulary entries can be
        // encountered in any order this condition does not hold true in this case.
        // However, keeping only one dense array at a time, greatly limits memory consumption
        // in sparse cases.
        let mut term_frequencies: Array1<usize> = Array1::zeros(self.vocabulary.len());
        let string = transform_string(document, &self.properties);
        let words = regex.find_iter(&string).map(|mat| mat.as_str()).collect();
        let list = NGramList::new(words, self.properties.n_gram_range);
        for ngram_items in list {
            for item in ngram_items {
                if let Some((item_index, _)) = self.vocabulary.get(&item) {
                    let term_freq = term_frequencies.get_mut(*item_index).unwrap();
                    *term_freq += 1;
                }
            }
        }
        let mut sprs_term_frequencies = CsVec::empty(self.vocabulary.len());

        // only insert non-zero elements in order to keep a sparse representation
        for (i, freq) in term_frequencies
            .into_iter()
            .enumerate()
            .filter(|(_, f)| **f > 0)
        {
            sprs_term_frequencies.append(i, *freq);
            doc_freqs[i] += 1;
        }
        sprs_term_frequencies
    }
}

fn validate_properties(properties: &VectorizerProperties) -> Result<Regex> {
    let (n_gram_min, n_gram_max) = properties.n_gram_range;
    if n_gram_min == 0 || n_gram_max == 0 {
        return Err(Error::InvalidNGramBoundaries(n_gram_min, n_gram_max));
    }
    if n_gram_min > n_gram_max {
        return Err(Error::FlippedNGramBoundaries(n_gram_min, n_gram_max));
    }
    let (min_freq, max_freq) = properties.document_frequency;
    if min_freq > 1. || max_freq < 0. {
        return Err(Error::InvalidDocumentFrequencies(min_freq, max_freq));
    }
    if max_freq < min_freq {
        return Err(Error::FlippedDocumentFrequencies(min_freq, max_freq));
    }
    Ok(Regex::new(&properties.split_regex)?)
}

fn transform_string(mut string: String, properties: &VectorizerProperties) -> String {
    if properties.normalize {
        string = string.nfkd().collect();
    }
    if properties.convert_to_lowercase {
        string = string.to_lowercase();
    }
    string
}

fn hashmap_to_vocabulary(map: &mut HashMap<String, (usize, usize)>) -> Vec<String> {
    let mut vec = Vec::with_capacity(map.len());
    for (word, (ref mut idx, _)) in map {
        *idx = vec.len();
        vec.push(word.clone());
    }
    vec
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::column_for_word;
    use ndarray::{array, Array2};
    use std::fs::File;
    use std::io::Write;

    macro_rules! assert_counts_for_word {

        ($voc:expr, $transf:expr, $(($word:expr, $counts:expr)),*) => {
            $ (
                assert_eq!(column_for_word!($voc, $transf, $word), $counts);
            )*
        }
    }

    #[test]
    fn simple_count_test() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::default().fit(&texts).unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).to_dense();
        let true_vocabulary = vec!["one", "two", "three", "four"];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("one", array![1, 0, 0, 0]),
            ("two", array![1, 1, 0, 0]),
            ("three", array![1, 1, 1, 0]),
            ("four", array![1, 1, 1, 1])
        );

        let vectorizer = CountVectorizer::default()
            .n_gram_range(2, 2)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).to_dense();
        let true_vocabulary = vec!["one two", "two three", "three four"];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("one two", array![1, 0, 0, 0]),
            ("two three", array![1, 1, 0, 0]),
            ("three four", array![1, 1, 1, 0])
        );

        let vectorizer = CountVectorizer::default()
            .n_gram_range(1, 2)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).to_dense();
        let true_vocabulary = vec![
            "one",
            "one two",
            "two",
            "two three",
            "three",
            "three four",
            "four",
        ];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("one", array![1, 0, 0, 0]),
            ("one two", array![1, 0, 0, 0]),
            ("two", array![1, 1, 0, 0]),
            ("two three", array![1, 1, 0, 0]),
            ("three", array![1, 1, 1, 0]),
            ("three four", array![1, 1, 1, 0]),
            ("four", array![1, 1, 1, 1])
        );
    }

    #[test]
    fn simple_count_test_vocabulary() {
        let texts = array![
            "apples.and.trees fi",
            "flowers,and,bees",
            "trees!here;and trees:there",
            "four bees and apples and apples again \u{FB01}"
        ];
        let vocabulary = ["apples", "bees", "flowers", "trees", "fi"];
        let vectorizer = CountVectorizer::default()
            .fit_vocabulary(&vocabulary)
            .unwrap();
        let vect_vocabulary = vectorizer.vocabulary();
        assert_vocabulary_eq(&vocabulary, vect_vocabulary);
        let transformed: Array2<usize> = vectorizer.transform(&texts).to_dense();
        assert_counts_for_word!(
            vect_vocabulary,
            transformed,
            ("apples", array![1, 0, 0, 2]),
            ("bees", array![0, 1, 0, 1]),
            ("flowers", array![0, 1, 0, 0]),
            ("trees", array![1, 0, 2, 0]),
            ("fi", array![1, 0, 0, 1])
        );
    }

    #[test]
    fn simple_count_no_punctuation_test() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::default()
            .split_regex(r"\b[^ ][^ ]+\b")
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).to_dense();
        let true_vocabulary = vec!["one", "two", "three", "four", "three;four"];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("one", array![1, 0, 0, 0]),
            ("two", array![1, 1, 0, 0]),
            ("three", array![1, 1, 0, 0]),
            ("four", array![1, 1, 0, 1]),
            ("three;four", array![0, 0, 1, 0])
        );
    }

    #[test]
    fn simple_count_no_lowercase_test() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::default()
            .convert_to_lowercase(false)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).to_dense();
        let true_vocabulary = vec!["oNe", "two", "three", "four", "TWO"];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("oNe", array![1, 0, 0, 0]),
            ("two", array![1, 0, 0, 0]),
            ("three", array![1, 1, 1, 0]),
            ("four", array![1, 1, 1, 1]),
            ("TWO", array![0, 1, 0, 0])
        );
    }

    #[test]
    fn simple_count_no_both_test() {
        let texts = array![
            "oNe oNe two three four",
            "TWO three four",
            "three;four",
            "four"
        ];
        let vectorizer = CountVectorizer::default()
            .convert_to_lowercase(false)
            .split_regex(r"\b[^ ][^ ]+\b")
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).to_dense();
        let true_vocabulary = vec!["oNe", "two", "three", "four", "TWO", "three;four"];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("oNe", array![2, 0, 0, 0]),
            ("two", array![1, 0, 0, 0]),
            ("three", array![1, 1, 0, 0]),
            ("four", array![1, 1, 0, 1]),
            ("TWO", array![0, 1, 0, 0]),
            ("three;four", array![0, 0, 1, 0])
        );
    }

    #[test]
    fn test_min_max_df() {
        let texts = array![
            "one and two and three",
            "three and four and five",
            "seven and eight",
            "maybe ten and eleven",
            "avoid singletons: one two four five seven eight ten eleven and an and"
        ];
        let vectorizer = CountVectorizer::default()
            .document_frequency(2. / 5., 3. / 5.)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let true_vocabulary = vec![
            "one", "two", "three", "four", "five", "seven", "eight", "ten", "eleven",
        ];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
    }

    #[test]
    fn test_fit_transform_files() {
        let text_files = create_test_files();
        let vectorizer = CountVectorizer::default()
            .fit_files(
                &text_files[..],
                encoding::all::UTF_8,
                encoding::DecoderTrap::Strict,
            )
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer
            .transform_files(
                &text_files[..],
                encoding::all::UTF_8,
                encoding::DecoderTrap::Strict,
            )
            .to_dense();
        let true_vocabulary = vec!["one", "two", "three", "four"];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("one", array![1, 0, 0, 0]),
            ("two", array![1, 1, 0, 0]),
            ("three", array![1, 1, 1, 0]),
            ("four", array![1, 1, 1, 1])
        );

        let vectorizer = CountVectorizer::default()
            .n_gram_range(2, 2)
            .fit_files(
                &text_files[..],
                encoding::all::UTF_8,
                encoding::DecoderTrap::Strict,
            )
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer
            .transform_files(
                &text_files[..],
                encoding::all::UTF_8,
                encoding::DecoderTrap::Strict,
            )
            .to_dense();
        let true_vocabulary = vec!["one two", "two three", "three four"];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("one two", array![1, 0, 0, 0]),
            ("two three", array![1, 1, 0, 0]),
            ("three four", array![1, 1, 1, 0])
        );

        let vectorizer = CountVectorizer::default()
            .n_gram_range(1, 2)
            .fit_files(
                &text_files[..],
                encoding::all::UTF_8,
                encoding::DecoderTrap::Strict,
            )
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer
            .transform_files(
                &text_files[..],
                encoding::all::UTF_8,
                encoding::DecoderTrap::Strict,
            )
            .to_dense();
        let true_vocabulary = vec![
            "one",
            "one two",
            "two",
            "two three",
            "three",
            "three four",
            "four",
        ];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("one", array![1, 0, 0, 0]),
            ("one two", array![1, 0, 0, 0]),
            ("two", array![1, 1, 0, 0]),
            ("two three", array![1, 1, 0, 0]),
            ("three", array![1, 1, 1, 0]),
            ("three four", array![1, 1, 1, 0]),
            ("four", array![1, 1, 1, 1])
        );
        delete_test_files(&text_files);
    }

    #[test]
    fn test_stopwords() {
        let texts = array![
            "one and two and three",
            "three and four and five",
            "seven and eight",
            "maybe ten and eleven",
            "avoid singletons: one two four five seven eight ten eleven and an and"
        ];
        let stopwords = ["and", "maybe", "an"];
        let vectorizer = CountVectorizer::default()
            .stopwords(&stopwords)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let true_vocabulary = vec![
            "one",
            "two",
            "three",
            "four",
            "five",
            "seven",
            "eight",
            "ten",
            "eleven",
            "avoid",
            "singletons",
        ];
        println!("voc: {:?}", vocabulary);
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
    }

    #[test]
    fn test_invalid_gram_boundaries() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::default().n_gram_range(0, 1).fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::default().n_gram_range(1, 0).fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::default().n_gram_range(2, 1).fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::default()
            .document_frequency(1.1, 1.)
            .fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::default()
            .document_frequency(1., -0.1)
            .fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::default()
            .document_frequency(0.5, 0.2)
            .fit(&texts);
        assert!(vectorizer.is_err());
    }

    #[test]
    fn test_invalid_regex() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::default().split_regex(r"[").fit(&texts);
        assert!(vectorizer.is_err())
    }

    fn assert_vocabulary_eq<T: ToString>(true_voc: &[T], voc: &[String]) {
        for word in true_voc {
            assert!(voc.contains(&word.to_string()));
        }
        assert_eq!(true_voc.len(), voc.len());
    }

    fn create_test_files() -> Vec<&'static str> {
        let file_names = vec![
            "./count_vectorization_test_file_1",
            "./count_vectorization_test_file_2",
            "./count_vectorization_test_file_3",
            "./count_vectorization_test_file_4",
        ];
        let contents = vec!["oNe two three four", "TWO three four", "three;four", "four"];
        //create files and write contents
        for (f_name, f_content) in file_names.iter().zip(contents.iter()) {
            let mut file = File::create(f_name).unwrap();
            file.write_all(f_content.as_bytes()).unwrap();
        }
        file_names
    }

    fn delete_test_files(file_names: &[&'static str]) {
        for f_name in file_names.iter() {
            std::fs::remove_file(f_name).unwrap();
        }
    }
}
