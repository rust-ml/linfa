/// Given a sequence of words, the queue can be iterated to obtain all the n-grams in the sequence,
/// starting from n-grams of lenght `min` up to n_grams of length `max`. The name "queue" is left from
/// a previous implementation but I left it because it sounded nice. Suggestions are welcome
pub struct NGramQueue<T: ToString> {
    min: usize,
    max: usize,
    queue: Vec<T>,
}

pub struct NGramQueueIntoIterator<T: ToString> {
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

#[macro_export]
macro_rules! column_for_word {
    ($voc:expr, $transf:expr, $word: expr ) => {
        $transf.column($voc.iter().position(|s| *s == $word.to_string()).unwrap())
    };
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
