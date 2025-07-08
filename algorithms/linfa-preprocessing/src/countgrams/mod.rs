//! Count vectorization methods

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::iter::IntoIterator;

use encoding::types::EncodingRef;
use encoding::DecoderTrap;
use itertools::sorted;
use ndarray::{Array1, ArrayBase, ArrayViewMut1, Data, Ix1};
use regex::Regex;
use sprs::{CsMat, CsVec};
use unicode_normalization::UnicodeNormalization;

use crate::error::{PreprocessingError, Result};
use crate::helpers::NGramList;
pub use hyperparams::{CountVectorizerParams, CountVectorizerValidParams};
use linfa::ParamGuard;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

mod hyperparams;

pub(crate) type Tokenizerfp = fn(&str) -> Vec<&str>;
pub enum Tokenizer {
    Function(Tokenizerfp),
    Regex(String),
}

impl CountVectorizerValidParams {
    /// Learns a vocabulary from the documents in `x`, according to the specified attributes and maps each
    /// vocabulary entry to an integer value, producing a [CountVectorizer](CountVectorizer).
    ///
    /// Returns an error if:
    /// * one of the `n_gram` boundaries is set to zero or the minimum value is greater than the maximum value
    /// * if the minimum document frequency is greater than one or than the maximum frequency, or if the maximum frequency is  
    ///   smaller than zero
    /// * if the regex expression for the split is invalid
    pub fn fit<T: ToString + Clone, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Result<CountVectorizer> {
        // word, (integer mapping for word, document frequency for word)
        let mut vocabulary: HashMap<String, (usize, usize)> = HashMap::new();
        for string in x.iter().map(|s| transform_string(s.to_string(), self)) {
            self.read_document_into_vocabulary(string, &self.split_regex(), &mut vocabulary);
        }

        let mut vocabulary = self.filter_vocabulary(vocabulary, x.len());

        let vec_vocabulary = hashmap_to_vocabulary(&mut vocabulary);

        Ok(CountVectorizer {
            vocabulary,
            vec_vocabulary,
            properties: self.clone(),
        })
    }

    /// Learns a vocabulary from the documents contained in the files in `input`, according to the specified attributes and maps each
    /// vocabulary entry to an integer value, producing a [CountVectorizer](CountVectorizer).
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
    ) -> Result<CountVectorizer> {
        // word, (integer mapping for word, document frequency for word)
        let mut vocabulary: HashMap<String, (usize, usize)> = HashMap::new();
        let documents_count = input.len();
        for path in input {
            let mut file = std::fs::File::open(path)?;
            let mut document_bytes = Vec::new();
            file.read_to_end(&mut document_bytes)?;
            let document = encoding::decode(&document_bytes, trap, encoding).0;
            // encoding error contains a cow string, can't just use ?, must go through the unwrap
            if document.is_err() {
                return Err(PreprocessingError::EncodingError(document.err().unwrap()));
            }
            // safe unwrap now that error has been handled
            let document = transform_string(document.unwrap(), self);
            self.read_document_into_vocabulary(document, &self.split_regex(), &mut vocabulary);
        }

        let mut vocabulary = self.filter_vocabulary(vocabulary, documents_count);
        let vec_vocabulary = hashmap_to_vocabulary(&mut vocabulary);

        Ok(CountVectorizer {
            vocabulary,
            vec_vocabulary,
            properties: self.clone(),
        })
    }

    /// Produces a [CountVectorizer](CountVectorizer) with the input vocabulary.
    /// All struct attributes are ignored in the fitting but will be used by the [CountVectorizer](CountVectorizer)
    /// to transform any text to be examined. As such this will return an error in the same cases as the `fit` method.
    pub fn fit_vocabulary<T: ToString>(&self, words: &[T]) -> Result<CountVectorizer> {
        let mut vocabulary: HashMap<String, (usize, usize)> = HashMap::with_capacity(words.len());
        for item in words.iter().map(|w| w.to_string()) {
            let len = vocabulary.len();
            // do not care about frequencies/stopwords if a vocabulary is given. Always 1 frequency
            vocabulary.entry(item).or_insert((len, 1));
        }
        let vec_vocabulary = hashmap_to_vocabulary(&mut vocabulary);
        Ok(CountVectorizer {
            vocabulary,
            vec_vocabulary,
            properties: self.clone(),
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
        let (min_df, max_df) = self.document_frequency();
        let len_f32 = n_documents as f32;
        let (min_abs_df, max_abs_df) = ((min_df * len_f32) as usize, (max_df * len_f32) as usize);

        let vocabulary = if min_abs_df == 0 && max_abs_df == n_documents {
            match &self.stopwords() {
                None => vocabulary,
                Some(stopwords) => vocabulary
                    .into_iter()
                    .filter(|(entry, (_, _))| !stopwords.contains(entry))
                    .collect(),
            }
        } else {
            match &self.stopwords() {
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
        };

        if let Some(max_features) = self.max_features() {
            sorted(
                vocabulary
                    .into_iter()
                    .map(|(word, (x, freq))| (Reverse(freq), Reverse(word), x)),
            )
            .take(max_features)
            .map(|(freq, word, x)| (word.0, (x, freq.0)))
            .collect()
        } else {
            vocabulary
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
        let words = if let Some(tokenizer) = self.tokenizer_function() {
            tokenizer(&doc)
        } else {
            regex.find_iter(&doc).map(|mat| mat.as_str()).collect()
        };
        let list = NGramList::new(words, self.n_gram_range());
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

impl CountVectorizerParams {
    /// Learns a vocabulary from the documents in `x`, according to the specified attributes and maps each
    /// vocabulary entry to an integer value, producing a [CountVectorizer](CountVectorizer).
    ///
    /// Returns an error if:
    /// * one of the `n_gram` boundaries is set to zero or the minimum value is greater than the maximum value
    /// * if the minimum document frequency is greater than one or than the maximum frequency, or if the maximum frequency is  
    ///   smaller than zero
    /// * if the regex expression for the split is invalid
    pub fn fit<T: ToString + Clone, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Result<CountVectorizer> {
        self.check_ref().and_then(|params| params.fit(x))
    }

    /// Learns a vocabulary from the documents contained in the files in `input`, according to the specified attributes and maps each
    /// vocabulary entry to an integer value, producing a [CountVectorizer](CountVectorizer).
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
    ) -> Result<CountVectorizer> {
        self.check_ref()
            .and_then(|params| params.fit_files(input, encoding, trap))
    }

    /// Produces a [CountVectorizer](CountVectorizer) with the input vocabulary.
    /// All struct attributes are ignored in the fitting but will be used by the [CountVectorizer](CountVectorizer)
    /// to transform any text to be examined. As such this will return an error in the same cases as the `fit` method.
    pub fn fit_vocabulary<T: ToString>(&self, words: &[T]) -> Result<CountVectorizer> {
        self.check_ref()
            .and_then(|params| params.fit_vocabulary(words))
    }
}

/// Counts the occurrences of each vocabulary entry, learned during fitting, in a sequence of documents. Each vocabulary entry is mapped
/// to an integer value that is used to index the count in the result.
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct CountVectorizer {
    pub(crate) vocabulary: HashMap<String, (usize, usize)>,
    pub(crate) vec_vocabulary: Vec<String>,
    pub(crate) properties: CountVectorizerValidParams,
}

impl CountVectorizer {
    /// Construct a new set of parameters
    pub fn params() -> CountVectorizerParams {
        CountVectorizerParams::default()
    }

    /// Number of vocabulary entries learned during fitting
    pub fn nentries(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn force_tokenizer_function_redefinition(&mut self, tokenizer: Tokenizerfp) {
        self.properties.tokenizer_function = Some(tokenizer);
    }

    pub(crate) fn validate_deserialization(&self) -> Result<()> {
        if self.properties.tokenizer_function().is_none()
            && self.properties.tokenizer_deserialization_guard
        {
            return Err(PreprocessingError::TokenizerNotSet);
        }

        Ok(())
    }

    /// Given a sequence of `n` documents, produces a sparse array of size `(n, vocabulary_entries)` where column `j` of row `i`
    /// is the number of occurrences of vocabulary entry `j` in the document of index `i`. Vocabulary entry `j` is the string
    /// at the `j`-th position in the vocabulary. If a vocabulary entry was not encountered in a document, then the relative
    /// cell in the sparse matrix will be set to `None`.
    pub fn transform<T: ToString, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Result<CsMat<usize>> {
        self.validate_deserialization()?;
        let (vectorized, _) = self.get_term_and_document_frequencies(x);
        Ok(vectorized)
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
    ) -> Result<CsMat<usize>> {
        self.validate_deserialization()?;
        let (vectorized, _) = self.get_term_and_document_frequencies_files(input, encoding, trap);
        Ok(vectorized)
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
        let regex = self.properties.split_regex();
        for string in x.into_iter().map(|s| s.to_string()) {
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
        let regex = self.properties.split_regex();
        for file_path in input.iter() {
            let mut file = std::fs::File::open(file_path).unwrap();
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
        let words = if let Some(tokenizer) = self.properties.tokenizer_function() {
            tokenizer(&string)
        } else {
            regex.find_iter(&string).map(|mat| mat.as_str()).collect()
        };
        let list = NGramList::new(words, self.properties.n_gram_range());
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
            .filter(|(_, f)| *f > 0)
        {
            sprs_term_frequencies.append(i, freq);
            doc_freqs[i] += 1;
        }
        sprs_term_frequencies
    }
}

fn transform_string(mut string: String, properties: &CountVectorizerValidParams) -> String {
    if properties.normalize() {
        string = string.nfkd().collect();
    }
    if properties.convert_to_lowercase() {
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
        let vectorizer = CountVectorizer::params().fit(&texts).unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).unwrap().to_dense();
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

        let vectorizer = CountVectorizer::params()
            .n_gram_range(2, 2)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).unwrap().to_dense();
        let true_vocabulary = vec!["one two", "two three", "three four"];
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
        assert_counts_for_word!(
            vocabulary,
            counts,
            ("one two", array![1, 0, 0, 0]),
            ("two three", array![1, 1, 0, 0]),
            ("three four", array![1, 1, 1, 0])
        );

        let vectorizer = CountVectorizer::params()
            .n_gram_range(1, 2)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).unwrap().to_dense();
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
        let vectorizer = CountVectorizer::params()
            .fit_vocabulary(&vocabulary)
            .unwrap();
        let vect_vocabulary = vectorizer.vocabulary();
        assert_vocabulary_eq(&vocabulary, vect_vocabulary);
        let transformed: Array2<usize> = vectorizer.transform(&texts).unwrap().to_dense();
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
        let vectorizer = CountVectorizer::params()
            .tokenizer(Tokenizer::Regex(r"\b[^ ][^ ]+\b".to_string()))
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).unwrap().to_dense();
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
        let vectorizer = CountVectorizer::params()
            .convert_to_lowercase(false)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts: Array2<usize> = vectorizer.transform(&texts).unwrap().to_dense();
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
        for vectorizer in [
            CountVectorizer::params()
                .convert_to_lowercase(false)
                .tokenizer(Tokenizer::Regex(r"\b[^ ][^ ]+\b".to_string()))
                .fit(&texts)
                .unwrap(),
            CountVectorizer::params()
                .convert_to_lowercase(false)
                .tokenizer(Tokenizer::Function(|x| x.split(" ").collect()))
                .fit(&texts)
                .unwrap(),
        ] {
            let vocabulary = vectorizer.vocabulary();
            let counts: Array2<usize> = vectorizer.transform(&texts).unwrap().to_dense();
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
        let vectorizer = CountVectorizer::params()
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
        let vectorizer = CountVectorizer::params()
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
            .unwrap()
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

        let vectorizer = CountVectorizer::params()
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
            .unwrap()
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

        let vectorizer = CountVectorizer::params()
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
            .unwrap()
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
        let vectorizer = CountVectorizer::params()
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
        println!("voc: {vocabulary:?}");
        assert_vocabulary_eq(&true_vocabulary, vocabulary);
    }

    #[test]
    fn test_invalid_gram_boundaries() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::params().n_gram_range(0, 1).fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::params().n_gram_range(1, 0).fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::params().n_gram_range(2, 1).fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::params()
            .document_frequency(1.1, 1.)
            .fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::params()
            .document_frequency(1., -0.1)
            .fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::params()
            .document_frequency(0.5, 0.2)
            .fit(&texts);
        assert!(vectorizer.is_err());
    }

    #[test]
    fn test_invalid_regex() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::params()
            .tokenizer(Tokenizer::Regex(r"[".to_string()))
            .fit(&texts);
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
        let contents = &["oNe two three four", "TWO three four", "three;four", "four"];
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
