use crate::error::{Error, Result};
use ndarray::{Array2, ArrayBase, Data, Ix1};
use std::collections::HashMap;
use std::iter::IntoIterator;

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
struct VectorizerProperties {
    remove_punctuation: bool,
    convert_to_lowercase: bool,
    punctuation_symbols: Vec<char>,
    n_gram_range: (usize, usize),
}

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
        self.properties.punctuation_symbols = punctuation_symbols.iter().map(|x| *x).collect();
        self
    }

    pub fn n_gram_range(mut self, n_gram_range: (usize, usize)) -> Self {
        self.properties.n_gram_range = n_gram_range;
        self
    }

    pub fn fit<T: ToString + Clone, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Result<FittedCountVectorizer> {
        validate_properties(&self.properties)?;

        let mut vocabulary: HashMap<String, usize> = HashMap::new();
        for string in x
            .iter()
            .map(|s| transform_string(s.to_string(), &self.properties))
        {
            let words = string.split_whitespace().map(|s| s.to_string()).collect();
            let queue = NGramQueue::new(words, self.properties.n_gram_range);
            for ngram_items in queue {
                for item in ngram_items {
                    let len = vocabulary.len();
                    vocabulary.entry(item).or_insert(len);
                }
            }
        }

        let vec_vocabulary = hashmap_to_vocabulary(&vocabulary);
        Ok(FittedCountVectorizer {
            vocabulary,
            vec_vocabulary,
            properties: self.properties.clone(),
        })
    }

    pub fn fit_vocabulary<T: ToString>(&self, words: &[T]) -> Result<FittedCountVectorizer> {
        let mut vocabulary: HashMap<String, usize> = HashMap::with_capacity(words.len());
        let words = words.into_iter().map(|w| w.to_string()).collect();
        let queue = NGramQueue::new(words, self.properties.n_gram_range);
        for ngram_items in queue {
            for item in ngram_items {
                let len = vocabulary.len();
                vocabulary.entry(item).or_insert(len);
            }
        }
        let vec_vocabulary = hashmap_to_vocabulary(&vocabulary);
        Ok(FittedCountVectorizer {
            vocabulary,
            vec_vocabulary,
            properties: self.properties.clone(),
        })
    }
}

pub struct FittedCountVectorizer {
    vocabulary: HashMap<String, usize>,
    vec_vocabulary: Vec<String>,
    properties: VectorizerProperties,
}

impl FittedCountVectorizer {
    pub fn transform<T: ToString, D: Data<Elem = T>>(
        &self,
        x: &ArrayBase<D, Ix1>,
    ) -> Array2<usize> {
        let mut vectorized = Array2::zeros((x.len(), self.vocabulary.len()));

        for (string_index, string) in x.into_iter().map(|s| s.to_string()).enumerate() {
            let string = transform_string(string, &self.properties);
            let words = string.split_whitespace().map(|s| s.to_string()).collect();
            let queue = NGramQueue::new(words, self.properties.n_gram_range);
            for ngram_items in queue {
                for item in ngram_items {
                    let item_index = self.vocabulary.get(&item);
                    if let Some(item_index) = item_index {
                        let value = vectorized.get_mut((string_index, *item_index)).unwrap();
                        *value += 1;
                    }
                }
            }
        }

        vectorized
    }

    pub fn vocabulary(&self) -> &Vec<String> {
        &self.vec_vocabulary
    }
}

fn validate_properties(properties: &VectorizerProperties) -> Result<()> {
    let (n_gram_min, n_gram_max) = properties.n_gram_range;
    if n_gram_min == 0 || n_gram_max == 0 {
        Err(Error::InvalidNGramBoundaries(n_gram_min, n_gram_max))
    } else if n_gram_min > n_gram_max {
        Err(Error::FlippedNGramBoundaries(n_gram_min, n_gram_max))
    } else {
        Ok(())
    }
}

fn transform_string(mut string: String, properties: &VectorizerProperties) -> String {
    if properties.remove_punctuation {
        string = string.replace(&properties.punctuation_symbols[..], " ")
    }
    if properties.convert_to_lowercase {
        string = string.to_lowercase();
    }
    string
}

fn hashmap_to_vocabulary(map: &HashMap<String, usize>) -> Vec<String> {
    let mut vec = vec![String::new(); map.len()];
    for (word, index) in map {
        vec[*index] = word.clone();
    }
    vec
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::array;

    #[test]
    fn simple_count_test() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::default().fit(&texts).unwrap();
        let vocabulary = vectorizer.vocabulary().clone();
        let counts = vectorizer.transform(&texts);
        assert_eq!(
            vocabulary,
            vec![
                "one".to_string(),
                "two".to_string(),
                "three".to_string(),
                "four".to_string()
            ]
        );
        assert_eq!(counts.row(0), array![1, 1, 1, 1]);
        assert_eq!(counts.row(1), array![0, 1, 1, 1]);
        assert_eq!(counts.row(2), array![0, 0, 1, 1]);
        assert_eq!(counts.row(3), array![0, 0, 0, 1]);

        let vectorizer = CountVectorizer::default()
            .n_gram_range((2, 2))
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary().clone();
        let counts = vectorizer.transform(&texts);
        assert_eq!(
            vocabulary,
            vec![
                "one two".to_string(),
                "two three".to_string(),
                "three four".to_string(),
            ]
        );
        assert_eq!(counts.row(0), array![1, 1, 1]);
        assert_eq!(counts.row(1), array![0, 1, 1]);
        assert_eq!(counts.row(2), array![0, 0, 1]);
        assert_eq!(counts.row(3), array![0, 0, 0]);

        let vectorizer = CountVectorizer::default()
            .n_gram_range((1, 2))
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary().clone();
        let counts = vectorizer.transform(&texts);
        assert_eq!(
            vocabulary,
            vec![
                "one".to_string(),
                "one two".to_string(),
                "two".to_string(),
                "two three".to_string(),
                "three".to_string(),
                "three four".to_string(),
                "four".to_string()
            ]
        );
        assert_eq!(counts.row(0), array![1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(counts.row(1), array![0, 0, 1, 1, 1, 1, 1]);
        assert_eq!(counts.row(2), array![0, 0, 0, 0, 1, 1, 1]);
        assert_eq!(counts.row(3), array![0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn simple_count_no_punctuation_test() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::default()
            .remove_punctuation(false)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary().clone();
        let counts = vectorizer.transform(&texts);
        assert_eq!(
            vocabulary,
            vec![
                "one".to_string(),
                "two".to_string(),
                "three".to_string(),
                "four".to_string(),
                "three;four".to_string()
            ]
        );
        assert_eq!(counts.row(0), array![1, 1, 1, 1, 0]);
        assert_eq!(counts.row(1), array![0, 1, 1, 1, 0]);
        assert_eq!(counts.row(2), array![0, 0, 0, 0, 1]);
        assert_eq!(counts.row(3), array![0, 0, 0, 1, 0]);
    }

    #[test]
    fn simple_count_no_lowercase_test() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::default()
            .convert_to_lowercase(false)
            .fit(&texts)
            .unwrap();
        let vocabulary = vectorizer.vocabulary().clone();
        let counts = vectorizer.transform(&texts);
        assert_eq!(
            vocabulary,
            vec![
                "oNe".to_string(),
                "two".to_string(),
                "three".to_string(),
                "four".to_string(),
                "TWO".to_string()
            ]
        );
        assert_eq!(counts.row(0), array![1, 1, 1, 1, 0]);
        assert_eq!(counts.row(1), array![0, 0, 1, 1, 1]);
        assert_eq!(counts.row(2), array![0, 0, 1, 1, 0]);
        assert_eq!(counts.row(3), array![0, 0, 0, 1, 0]);
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
        let vocabulary = vectorizer.vocabulary().clone();
        let counts = vectorizer.transform(&texts);
        assert_eq!(
            vocabulary,
            vec![
                "oNe".to_string(),
                "two".to_string(),
                "three".to_string(),
                "four".to_string(),
                "TWO".to_string(),
                "three;four".to_string()
            ]
        );
        assert_eq!(counts.row(0), array![2, 1, 1, 1, 0, 0]);
        assert_eq!(counts.row(1), array![0, 0, 1, 1, 1, 0]);
        assert_eq!(counts.row(2), array![0, 0, 0, 0, 0, 1]);
        assert_eq!(counts.row(3), array![0, 0, 0, 1, 0, 0]);
    }
    #[test]
    fn test_ngram_queue() {
        let words = vec![
            "oNe".to_string(),
            "oNe".to_string(),
            "two".to_string(),
            "three".to_string(),
            "four".to_string(),
            "TWO".to_string(),
            "three".to_string(),
            "four".to_string(),
            "three;four".to_string(),
            "four".to_string(),
        ];
        let queue = NGramQueue::new(words.clone(), (1, 1));
        for (i, items) in queue.into_iter().enumerate() {
            assert_eq!(items.len(), 1);
            assert_eq!(items[0], words[i].clone());
        }

        let queue = NGramQueue::new(words.clone(), (2, 2));
        for (i, items) in queue.into_iter().enumerate() {
            assert_eq!(items.len(), 1);
            assert_eq!(items[0], words[i].clone() + " " + words[i + 1].as_str());
        }
        let queue = NGramQueue::new(words.clone(), (1, 2));
        for (i, items) in queue.into_iter().enumerate() {
            if i < words.len() - 1 {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], words[i].clone());
                assert_eq!(items[1], words[i].clone() + " " + words[i + 1].as_str());
            } else {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0], words[i].clone());
            }
        }
    }

    #[test]
    fn test_invalid_gram_boundaries() {
        let texts = array!["oNe two three four", "TWO three four", "three;four", "four"];
        let vectorizer = CountVectorizer::default().n_gram_range((0, 1)).fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::default().n_gram_range((1, 0)).fit(&texts);
        assert!(vectorizer.is_err());
        let vectorizer = CountVectorizer::default().n_gram_range((2, 1)).fit(&texts);
        assert!(vectorizer.is_err());
    }
}
