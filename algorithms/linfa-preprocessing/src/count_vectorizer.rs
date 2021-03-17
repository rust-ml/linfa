//! Count vectorization methods

use crate::error::{Error, Result};
use ndarray::{Array2, ArrayBase, Data, Ix1};
use std::collections::{HashMap, HashSet};
use std::iter::IntoIterator;
use unicode_normalization::UnicodeNormalization;

/// Given a sequence of words, the queue can be iterated to obtain all the n-grams in the sequence,
/// starting from n-grams of lenght `min` up to n_grams of length `max`. The name "queue" is left from
/// a previous implementation but I left it because it sounded nice. Suggestions are welcome
struct NGramQueue<T: ToString> {
    min: usize,
    max: usize,
    queue: Vec<T>,
}

struct NGramQueueIntoIterator<T: ToString> {
    queue: NGramQueue<T>,
    index: usize,
}

impl<T: ToString> Iterator for NGramQueueIntoIterator<T> {
    type Item = Vec<String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.queue.len() {
            return None;
        }
        let res = self.queue.ngram_items(self.index);
        if res.is_some() {
            self.index += 1;
            res
        } else {
            None
        }
    }
}

impl<T: ToString> IntoIterator for NGramQueue<T> {
    type Item = Vec<String>;
    type IntoIter = NGramQueueIntoIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        NGramQueueIntoIterator {
            queue: self,
            index: 0,
        }
    }
}

impl<T: ToString> NGramQueue<T> {
    pub fn new(vec: Vec<T>, range: (usize, usize)) -> Self {
        Self {
            min: range.0,
            max: range.1,
            queue: vec,
        }
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Constructs all n-grams obtainable from the word sequence starting from the word at `index`
    pub fn ngram_items(&self, index: usize) -> Option<Vec<String>> {
        let mut items = Vec::new();
        let len = self.queue.len();
        let min_end = index + self.min;
        if min_end > len {
            return None;
        }
        let max_end = if (index + self.max) < len {
            index + self.max
        } else {
            len
        };
        let mut item = self.queue[index].to_string();
        for j in (index + 1)..min_end {
            item.push_str(" ");
            item.push_str(&self.queue[j].to_string());
        }
        items.push(item.clone());
        for j in min_end..max_end {
            item.push_str(" ");
            item.push_str(&self.queue[j].to_string());
            items.push(item.clone())
        }
        Some(items)
    }
}

#[derive(Clone)]
/// Struct that holds all vectorizer options so that they can be passed to the fitted vectorizer
struct VectorizerProperties {
    remove_punctuation: bool,
    convert_to_lowercase: bool,
    punctuation_symbols: Vec<char>,
    n_gram_range: (usize, usize),
    normalize: bool,
    document_frequency: (f32, f32),
}

/// Count vectorizer: learns a vocabulary from a sequence of texts and maps each
/// vocabulary entry to an integer value, producing a [FittedCountVectorizer](struct.FittedCountVectorizer.html) that can
/// be used to count the occurrences of each vocabulary entry in any sequence of texts. Alternatively a user-specified vocabulary can
/// be used for fitting.
///
/// ### Attributes
///
/// If a user-defined vocabulary is used for fitting then the following attributes will not be considered during the fitting phase but
/// but they will still be used by the [FittedCountVectorizer](struct.FittedCountVectorizer.html) to transform any text to be examined.
///
/// * `remove_punctuation`: if true, punctuation symbols will be substituted by a whitespace in all texts used for fitting. Defaults to `true`
/// * `punctuation_symbols`: the list of punctuation sybols to be substituted by whitespace. The default list is: `['.', ',', ';', ':', '?', '!']`
/// * `convert_to_lowercase`: if true, all texts used for fitting will be converted to lowercase. Defaults to `true`.
/// * `n_gram_range`: if set to `(1,1)` single words will be candidate vocabulary entries, if `(2,2)` then adjacent words pairs will be considered,
///    if `(1,2)` then both single words and adjacent word pairs will be considered, and so on. The default value is `(1,1)`.
/// * `normalize`: if true, all charachters in the texts used for fitting will be normalized according to unicode's NFKD normalization. Defaults to `true`.
/// * `document_frequency`: specifies the minimum and maximum (relative) document frequencies that each vocabulary entry. Defaults to `(0., 1.)` (0% minimum and 100% maximum)
///
pub struct CountVectorizer {
    properties: VectorizerProperties,
}

impl std::default::Default for CountVectorizer {
    fn default() -> Self {
        Self {
            properties: VectorizerProperties {
                remove_punctuation: true,
                convert_to_lowercase: true,
                punctuation_symbols: vec!['.', ',', ';', ':', '?', '!'],
                n_gram_range: (1, 1),
                normalize: true,
                document_frequency: (0., 1.),
            },
        }
    }
}

impl CountVectorizer {
    pub fn remove_punctuation(mut self, remove_punctuation: bool) -> Self {
        self.properties.remove_punctuation = remove_punctuation;
        self
    }

    pub fn convert_to_lowercase(mut self, convert_to_lowercase: bool) -> Self {
        self.properties.convert_to_lowercase = convert_to_lowercase;
        self
    }

    pub fn punctuation_symbols(mut self, punctuation_symbols: &[char]) -> Self {
        self.properties.punctuation_symbols = punctuation_symbols.iter().copied().collect();
        self
    }

    pub fn n_gram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.properties.n_gram_range = (min_n, max_n);
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.properties.normalize = normalize;
        self
    }

    pub fn document_frequency(mut self, min_freq: f32, max_freq: f32) -> Self {
        self.properties.document_frequency = (min_freq, max_freq);
        self
    }

    /// Learns a vocabulary from the texts in `x`, according to the specified attributes and maps each
    /// vocabulary entry to an integer value, producing a [FittedCountVectorizer](struct.FittedCountVectorizer.html).
    ///
    /// Returns an error if:
    /// * one of the `n_gram` boundaries is set to zero or the minimum value is greater than the maximum value
    /// * if the minimum document frequency is greater than one or than the maximum frequency, or if the maximum frequecy is  
    ///   smaller than zero
    pub fn fit<T: ToString + Clone, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Result<FittedCountVectorizer> {
        validate_properties(&self.properties)?;

        // word, (integer mapping for word, document frequency for word)
        let mut vocabulary: HashMap<String, (usize, usize)> = HashMap::new();
        for string in x
            .iter()
            .map(|s| transform_string(s.to_string(), &self.properties))
        {
            let mut document_vocabulary: HashSet<String> = HashSet::new();
            let words = string.split_whitespace().collect();
            let queue = NGramQueue::new(words, self.properties.n_gram_range);
            for ngram_items in queue {
                for item in ngram_items {
                    // if item was already in the hashet it simply gets overwritten,
                    // not a problem
                    document_vocabulary.insert(item);
                }
            }
            for word in document_vocabulary {
                let len = vocabulary.len();
                if let Some((_, freq)) = vocabulary.get_mut(&word) {
                    *freq += 1;
                } else {
                    vocabulary.insert(word, (len, 1));
                }
            }
        }

        let len_f32 = x.len() as f32;
        let (min_abs_df, max_abs_df) = (
            (self.properties.document_frequency.0 * len_f32) as usize,
            (self.properties.document_frequency.1 * len_f32) as usize,
        );

        let mut vocabulary = vocabulary
            .into_iter()
            .filter(|(_, (_, abs_count))| *abs_count >= min_abs_df && *abs_count <= max_abs_df)
            .collect();

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
}

/// Counts the occurrences of each vocabulary entry, learned during fitting, in a sequence of texts. Each vocabulary entry is mapped
/// to an integer value that is used to index the count in the result.
pub struct FittedCountVectorizer {
    vocabulary: HashMap<String, (usize, usize)>,
    vec_vocabulary: Vec<String>,
    properties: VectorizerProperties,
}

impl FittedCountVectorizer {
    /// Given a sequence of `n` texts, produces an array of size `(n, vocabulary_entries)` where column `j` of row `i`
    /// is the number of occurrences of vocabulary entry `j` in the text of index `i`. Vocabulary entry `j` is the string
    /// at the `j`-th position in the vocabulary.
    pub fn transform<T: ToString, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Array2<usize> {
        let mut vectorized = Array2::zeros((x.len(), self.vocabulary.len()));
        for (string_index, string) in x.into_iter().map(|s| s.to_string()).enumerate() {
            let string = transform_string(string, &self.properties);
            let words = string.split_whitespace().collect();
            let queue = NGramQueue::new(words, self.properties.n_gram_range);
            for ngram_items in queue {
                for item in ngram_items {
                    if let Some((item_index, _)) = self.vocabulary.get(&item) {
                        let value = vectorized.get_mut((string_index, *item_index)).unwrap();
                        *value += 1;
                    }
                }
            }
        }

        vectorized
    }

    /// Constains all vocabulary entries, in the same order used by the `transform` method.
    pub fn vocabulary(&self) -> &Vec<String> {
        &self.vec_vocabulary
    }
}

fn validate_properties(properties: &VectorizerProperties) -> Result<()> {
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
    Ok(())
}

fn transform_string(mut string: String, properties: &VectorizerProperties) -> String {
    if properties.remove_punctuation {
        string = string.replace(&properties.punctuation_symbols[..], " ")
    }
    if properties.convert_to_lowercase {
        string = string.to_lowercase();
    }
    if properties.normalize {
        string = string.nfkd().collect();
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
    use ndarray::array;

    macro_rules! column_for_word {
        ($voc:expr, $transf:expr, $word: expr ) => {
            $transf.column($voc.iter().position(|s| *s == $word.to_string()).unwrap())
        };
    }

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
        let counts = vectorizer.transform(&texts);
        let true_vocabulary = vec!["one", "two", "three", "four"];
        assert_vocabulary_eq(&true_vocabulary, &vocabulary);
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
        let counts = vectorizer.transform(&texts);
        let true_vocabulary = vec!["one two", "two three", "three four"];
        assert_vocabulary_eq(&true_vocabulary, &vocabulary);
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
        let counts = vectorizer.transform(&texts);
        let true_vocabulary = vec![
            "one",
            "one two",
            "two",
            "two three",
            "three",
            "three four",
            "four",
        ];
        assert_vocabulary_eq(&true_vocabulary, &vocabulary);
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
        assert_vocabulary_eq(&vocabulary, &vect_vocabulary);
        let transformed = vectorizer.transform(&texts);
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
            .remove_punctuation(false)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts = vectorizer.transform(&texts);
        let true_vocabulary = vec!["one", "two", "three", "four", "three;four"];
        assert_vocabulary_eq(&true_vocabulary, &vocabulary);
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
        let counts = vectorizer.transform(&texts);
        let true_vocabulary = vec!["oNe", "two", "three", "four", "TWO"];
        assert_vocabulary_eq(&true_vocabulary, &vocabulary);
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
            .remove_punctuation(false)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary();
        let counts = vectorizer.transform(&texts);
        let true_vocabulary = vec!["oNe", "two", "three", "four", "TWO", "three;four"];
        assert_vocabulary_eq(&true_vocabulary, &vocabulary);
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
        assert_vocabulary_eq(&true_vocabulary, &vocabulary);
    }

    #[test]
    fn test_ngram_queue() {
        let words = vec![
            "oNe",
            "oNe",
            "two",
            "three",
            "four",
            "TWO",
            "three",
            "four",
            "three;four",
            "four",
        ];
        let queue = NGramQueue::new(words.clone(), (1, 1));
        for (i, items) in queue.into_iter().enumerate() {
            assert_eq!(items.len(), 1);
            assert_eq!(items[0], words[i].clone());
        }

        let queue = NGramQueue::new(words.clone(), (2, 2));
        for (i, items) in queue.into_iter().enumerate() {
            assert_eq!(items.len(), 1);
            assert_eq!(items[0], words[i].to_string() + " " + words[i + 1]);
        }
        let queue = NGramQueue::new(words.clone(), (1, 2));
        for (i, items) in queue.into_iter().enumerate() {
            if i < words.len() - 1 {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], words[i]);
                assert_eq!(items[1], words[i].to_string() + " " + words[i + 1]);
            } else {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0], words[i]);
            }
        }
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

    fn assert_vocabulary_eq<T: ToString>(true_voc: &[T], voc: &[String]) {
        for word in true_voc {
            assert!(voc.contains(&word.to_string()));
        }
        assert_eq!(true_voc.len(), voc.len());
    }
}
