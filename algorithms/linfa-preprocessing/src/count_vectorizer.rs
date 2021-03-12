use ndarray::{Array2, ArrayBase, Data, Ix1};
use std::collections::HashMap;
use std::iter::IntoIterator;

pub struct CountVectorizer {
    remove_punctuation: bool,
    convert_to_lowercase: bool,
    punctuation_symbols: Vec<char>,
}

impl std::default::Default for CountVectorizer {
    fn default() -> Self {
        Self {
            remove_punctuation: true,
            convert_to_lowercase: true,
            punctuation_symbols: vec!['.', ',', ';', ':'],
        }
    }
}

impl CountVectorizer {
    pub fn remove_punctuation(mut self, remove_punctuation: bool) -> Self {
        self.remove_punctuation = remove_punctuation;
        self
    }

    pub fn convert_to_lowercase(mut self, convert_to_lowercase: bool) -> Self {
        self.convert_to_lowercase = convert_to_lowercase;
        self
    }

    pub fn punctuation_symbols(mut self, punctuation_symbols: &[char]) -> Self {
        self.punctuation_symbols = punctuation_symbols.iter().map(|x| *x).collect();
        self
    }

    pub fn fit<D: Data<Elem = String>>(&self, x: &ArrayBase<D, Ix1>) -> FittedCountVectorizer {
        let mut vocabulary: HashMap<String, usize> = HashMap::new();
        for string in x.iter() {
            let string = transform_string(
                string.clone(),
                self.remove_punctuation,
                self.convert_to_lowercase,
                &self.punctuation_symbols,
            );
            for word in string.split_whitespace().map(|w| w.to_string()) {
                let len = vocabulary.len();
                vocabulary.entry(word).or_insert(len);
            }
        }
        let vec_vocabulary = hashmap_to_vocabulary(&vocabulary);
        FittedCountVectorizer {
            vocabulary,
            vec_vocabulary,
            remove_punctuation: self.remove_punctuation,
            convert_to_lowercase: self.convert_to_lowercase,
            punctuation_symbols: self.punctuation_symbols.clone(),
        }
    }

    pub fn fit_vocabulary<T: ToString>(&self, words: &[T]) -> FittedCountVectorizer {
        let mut vocabulary: HashMap<String, usize> = HashMap::with_capacity(words.len());
        for word in words.iter().map(|w| w.to_string()) {
            let len = vocabulary.len();
            vocabulary.entry(word).or_insert(len);
        }
        let vec_vocabulary = hashmap_to_vocabulary(&vocabulary);
        FittedCountVectorizer {
            vocabulary,
            vec_vocabulary,
            remove_punctuation: self.remove_punctuation,
            convert_to_lowercase: self.convert_to_lowercase,
            punctuation_symbols: self.punctuation_symbols.clone(),
        }
    }
}

pub struct FittedCountVectorizer {
    vocabulary: HashMap<String, usize>,
    vec_vocabulary: Vec<String>,
    remove_punctuation: bool,
    convert_to_lowercase: bool,
    punctuation_symbols: Vec<char>,
}

impl FittedCountVectorizer {
    pub fn transform<D: Data<Elem = String>>(&self, x: &ArrayBase<D, Ix1>) -> Array2<usize> {
        let mut vectorized = Array2::zeros((x.len(), self.vocabulary.len()));
        for (string_index, string) in x.into_iter().map(|s| s.clone()).enumerate() {
            let string = transform_string(
                string,
                self.remove_punctuation,
                self.convert_to_lowercase,
                &self.punctuation_symbols,
            );
            for word in string.split_whitespace().map(|w| w.to_string()) {
                let word_index = self.vocabulary.get(&word);
                if let Some(word_index) = word_index {
                    let value = vectorized.get_mut((string_index, *word_index)).unwrap();
                    *value += 1;
                }
            }
        }
        vectorized
    }

    pub fn vocabulary(&self) -> &Vec<String> {
        &self.vec_vocabulary
    }
}

fn transform_string(
    mut string: String,
    remove_punctuation: bool,
    convert_to_lowercase: bool,
    punctuation_symbols: &Vec<char>,
) -> String {
    if remove_punctuation {
        string = string.replace(&punctuation_symbols[..], " ")
    }
    if convert_to_lowercase {
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
        let texts = array![
            "oNe two three four".to_string(),
            "TWO three four".to_string(),
            "three;four".to_string(),
            "four".to_string()
        ];
        let vectorizer = CountVectorizer::default().fit(&texts);
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
    }

    #[test]
    fn simple_count_no_punctuation_test() {
        let texts = array![
            "oNe two three four".to_string(),
            "TWO three four".to_string(),
            "three;four".to_string(),
            "four".to_string()
        ];
        let vectorizer = CountVectorizer::default()
            .remove_punctuation(false)
            .fit(&texts);
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
        let texts = array![
            "oNe two three four".to_string(),
            "TWO three four".to_string(),
            "three;four".to_string(),
            "four".to_string()
        ];
        let vectorizer = CountVectorizer::default()
            .convert_to_lowercase(false)
            .fit(&texts);
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
            "oNe oNe two three four".to_string(),
            "TWO three four".to_string(),
            "three;four".to_string(),
            "four".to_string()
        ];
        let vectorizer = CountVectorizer::default()
            .convert_to_lowercase(false)
            .remove_punctuation(false)
            .fit(&texts);
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
}
