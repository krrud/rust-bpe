use std::collections::{HashMap, HashSet, VecDeque};
use rayon::prelude::*;
use std::io::stdout;
use std::io::{self, Write};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use serde_json;

use crate::tokenizer::TokenConfig;


#[derive(Serialize, Deserialize, Debug)]
pub struct Tokenizer {
    vocabulary: HashSet<String>,
    merge_rules: Vec<(String, String)>,
    token_to_index: HashMap<String, usize>,
    index_to_token: HashMap<usize, String>,
    config: TokenConfig,
}

impl Tokenizer {
    pub fn new(vocabulary: HashSet<String>, merge_rules: Vec<(String, String)>, config: TokenConfig) -> Self {
        let mut vocabulary = vocabulary;
        vocabulary.insert("<unk>".to_string());

        let mut token_to_index = HashMap::new();
        let mut index_to_token = HashMap::new();
        
        for (i, token) in vocabulary.iter().enumerate() {
            token_to_index.insert(token.clone(), i);
            index_to_token.insert(i, token.clone());
        }

        Tokenizer {
            vocabulary,
            merge_rules,
            token_to_index,
            index_to_token,
            config,
        }
    }

    pub fn get_vocabulary(&self) -> HashSet<String> {
        self.vocabulary.clone()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn get_merge_rules(&self) -> Vec<(String, String)> {
        self.merge_rules.clone()
    }

    pub fn get_token(&self, index: usize) -> Option<String> {
        self.index_to_token.get(&index).cloned()
    }

    pub fn get_index(&self, token: &str) -> Option<usize> {
        self.token_to_index.get(token).copied()
    }

    pub fn get_tokens(&self, indices: &Vec<usize>) -> Vec<String> {
        indices.iter().map(|&idx| self.get_token(idx).unwrap()).collect()
    }
    
    pub fn get_indices(&self, tokens: &Vec<String>) -> Vec<usize> {
        tokens.iter().map(|token| self.get_index(token).unwrap()).collect()
    }

    pub fn split_with_punctuation(text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut start = 0;
    
        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                // Include the punctuation mark in the sentence
                let end = i + c.len_utf8();
                result.push(text[start..end].to_string());
                // Move the start to the next character after the punctuation mark
                start = end;
            }
        }
    
        // Add any remaining text as the last sentence
        if start < text.len() {
            result.push(text[start..].to_string());
        }
    
        result
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // Tokenize text using trained bpe tokenizer
        let cleaned_text = Self::clean_text(text);
        let mut tokens = Vec::new();

        for sentence in Self::split_with_punctuation(&cleaned_text) {
            tokens.extend(self.tokenize_sentence(sentence.trim()));
        }

        tokens
    }

    pub fn tokenize_sentence(&self, sentence: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let mut start = 0;

        while start < sentence.len() {
            let mut max_match = "";
            let mut max_match_length = 0;

            // Find largest token that matches the text
            for end in start + 1..=sentence.len() {
                let substr = &sentence[start..end];
                if self.vocabulary.contains(substr) && substr.len() > max_match_length {
                    max_match = substr;
                    max_match_length = substr.len();
                }
            }

            if max_match.is_empty() {
                // If no token is found, append unknown token
                tokens.push(self.config.unknown.index);
                start += 1;
            } else {
                // Add the longest match and update the start position
                tokens.push(self.get_index(max_match).unwrap());
                start += max_match_length;
            }

            // Add end-of-word token if we are at the end of a word
            if start < sentence.len() && sentence.as_bytes()[start] == b' ' {
                tokens.push(self.config.space.index);
                start += 1; // Skip the space
            }
        }

        // Add end-of-sentence token
        if start == sentence.len() {
            tokens.push(self.config.eos.index);
        }

        tokens
    }

    pub fn detokenize(&self, indices: &[usize]) -> String {
        // Detokenize indices into a single string
        let mut result = String::new();
        for &idx in indices {
            if let Some(token) = self.get_token(idx) {
                if !token.trim().is_empty() || token == " "{
                    if self.config.is_eos(&token) {
                        result.push_str(" ");
                    } else {
                        result.push_str(&token);
                    }
                }
            }
        }
        result
    }

    pub fn train_cpu(source: &str, iterations: usize, output_filepath: &str, start_filepath: Option<&str>) -> Self {
        // Train tokenizer on CPU using byte pair encoding
        let mut config = TokenConfig::new();
        let mut merge_rules: Vec<(String, String)>;
        let mut token_list: Vec<String>;
        let mut token_indices: Vec<usize>;
        let mut token_map: HashMap<String, usize>;

        // Load existing tokenizer if provided
        if let Some(start) = start_filepath {
            let tokenizer = Tokenizer::load(start).expect("Failed to load tokenizer");
            merge_rules = tokenizer.get_merge_rules();
            token_list = tokenizer.get_vocabulary().into_iter().collect();

            // Rebuild token_map and token_indices using only space to separate words
            token_map = tokenizer.token_to_index.clone();
            token_indices = source.split_whitespace().flat_map(|word| {
                word.chars().map(|c| token_map.get(&c.to_string()).unwrap().to_owned()).chain(Some(config.space.index))
            }).collect();
            token_indices.pop(); 
        } else {
            // Initialize tokenizer from scratch
            merge_rules = Vec::new();
            token_list = config.get_values().iter().map(|v| v.to_string()).collect();

            // Split the source into words and handle end of words
            token_indices = Vec::with_capacity(source.len());
            token_map = HashMap::new();
            let values = config.get_values();
            let indices = config.get_indices();
            for (value, &index) in values.iter().zip(indices.iter()) {
                token_map.insert(value.to_string(), index);
            }

            for word in source.split_whitespace() {
                for c in word.chars() {
                    let s = c.to_string().to_lowercase();
                    let index = *token_map.entry(s.clone()).or_insert_with(|| {
                        token_list.push(s);
                        token_list.len() - 1
                    });
                    token_indices.push(index);
                }
                token_indices.push(config.space.index); // </w> between words
            }
            if *token_indices.last().unwrap() == config.space.index {
                token_indices.pop();
            }
        }

        let start_time = Instant::now();
        for i in 0..iterations {
            let iter_time = Instant::now();
    
            // Count token-pair frequencies
            let pair_frequencies = token_indices.par_windows(2)
                .fold(HashMap::new, |mut acc, window| {
                    if let [a, b] = window {
                        *acc.entry((*a, *b)).or_insert(0) += 1;
                    }
                    acc
                })
                .reduce(HashMap::new, |mut acc, partial| {
                    for (key, count) in partial {
                        *acc.entry(key).or_insert(0) += count;
                    }
                    acc
                });
    
            // Delay merging common pairs for first few iterations to avoid bottlenecks
            let delay_common_pairs = 10;
            let should_delay = |token: &String| token == " ";
    
            let (first, second) = if i < delay_common_pairs {
                pair_frequencies.iter()
                    .filter(|&(&(first, second), _)| {
                        !should_delay(&token_list[first]) && !should_delay(&token_list[second])
                    })
                    .max_by_key(|&(_, &count)| count)
            } else {
                pair_frequencies.iter().max_by_key(|&(_, &count)| count)
            }
            .map(|(&pair, _)| pair)
            .unwrap_or((usize::MAX, usize::MAX));
    
            // Find the most frequent pair and merge them
            if first != usize::MAX && second != usize::MAX {
                let new_token = format!("{}{}", token_list[first], token_list[second]);
                let new_index = token_list.len();
                token_list.push(new_token.clone());
                merge_rules.push((token_list[first].clone(), token_list[second].clone()));
    
                // Update token_indices
                let mut new_token_indices = Vec::with_capacity(token_indices.len());
                let mut skip = false;
                for window in token_indices.windows(2) {
                    if skip {
                        skip = false;
                        continue;
                    }
                    if window[0] == first && window[1] == second {
                        new_token_indices.push(new_index);
                        skip = true;
                    } else {
                        new_token_indices.push(window[0]);
                    }
                }
                if !skip {
                    if let Some(&last) = token_indices.last() {
                        new_token_indices.push(last);
                    }
                }
                token_indices = new_token_indices;
            } else {
                break; // No more pairs to merge - we're done here folks
            }
    
            // Save every 50 iterations
            if i % 50 == 0 {
                let vocabulary: HashSet<String> = token_list.clone().into_iter().collect();
                config.set_indices(Self::extract_indices(&vocabulary, &config));
                let tokenizer = Tokenizer::new(vocabulary, merge_rules.clone(), config.clone());
                tokenizer.save(output_filepath).unwrap();
            }
    
            let elapsed_time = start_time.elapsed().as_secs_f32();
            let percent = 100.0 * ((i + 1) as f32 / iterations as f32);
            let iteration_time = elapsed_time / (i as f32 + 1.0);
            println!("Iteration {} time: {:?}", i, iter_time.elapsed().as_secs_f32());
            io::stdout().flush().unwrap();
        }
    
        let vocabulary: HashSet<String> = token_list.into_iter().collect();
        config.set_indices(Self::extract_indices(&vocabulary, &config));
        let trained_tokenizer = Tokenizer::new(vocabulary, merge_rules, config);
        trained_tokenizer.save(output_filepath).unwrap();

        trained_tokenizer
    } 
 
    fn extract_indices(token_list: &HashSet<String>, config: &TokenConfig) -> Vec<usize> {
        let mut indices = Vec::new();
        for token in config.get_values() {
            // get index from hashset
            let index = token_list.iter().position(|t| t == &token).unwrap();
            indices.push(index);
        }
        indices
    }

    pub fn clean_text(text: &str) -> String {
        // Clean text by converting to lowercase and removing newlines
        let mut text = text.to_lowercase();
        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ");
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    pub fn process_dataset(dir: &str) -> String {
        // Process all .txt files in the provided directory into a single string
        let mut text = String::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let file = entry.unwrap();
            let path = file.path();
            if path.is_file() {
                println!("Reading file: {:?}", file.file_name());
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        let source_text = String::from_utf8_lossy(&bytes);
                        let clean_text = Tokenizer::clean_text(&source_text);
                        text.push(' ');
                        text.push_str(&clean_text);
                    },
                    Err(e) => {
                        eprintln!("Could not read file {:?}: {}", file.file_name(), e);
                        continue;
                    }
                }
            }
        }
        text
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        // Save tokenizer to a JSON file
        let json = serde_json::to_string(&self)?;
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        // Load tokenizer from a JSON file
        let data = std::fs::read_to_string(path)?;
        let config = TokenConfig::new();
        let tokenizer: Tokenizer = serde_json::from_str(&data)?;
        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // const TEST_STR: &str = "Hello, world! The quick brown fox jumps over the lazy dog 123 times. Can you believe it? This string features (parentheses), [brackets], {braces}, and even <angle brackets>. Don't forget about quotes: 'single', \"double\", and `backticks` for code. Special characters like @, #, $, %, &, *, ^, +, =, and | are also included.";
    const TEST_STR: &str = "In this example:

The create_vocab function creates an IndexMap from a list of tokens. Each token is associated with its index, but because IndexMap does not allow duplicates, repeated tokens are ignored after their first insertion.
The order of tokens as they first appear is preserved in the IndexMap, and each token can be accessed in O(1) average time complexity.
Conclusion
IndexMap is an excellent choice if you need the characteristics of both a map and a vector. It's particularly useful when the order of elements is significant and you also need fast access based on keys. This makes it a powerful tool for many scenarios in data processing, caching, and more, where both order and performance are crucial.";
    // const TEST_STR: &str = "we had a good time";
    #[test]
    fn validate_tokenizer() {
        let tokenizer = Tokenizer::load("./src/tokenizer_train.json").unwrap();
        let duplicates = has_duplicates(tokenizer.vocabulary.iter().cloned().collect::<Vec<_>>());
        assert_eq!(duplicates, false);

        let tokens = tokenizer.tokenize(&TEST_STR);
        let token_vals = tokenizer.get_tokens(&tokens);
        let token_indices = tokenizer.get_indices(&token_vals);
        let detokenized = tokenizer.detokenize(&tokens);

        println!("Input: {:?}", TEST_STR);
        println!("Tokens: {:?}", token_vals); 
        // println!("Token indices: {:?}", token_indices);
        println!("Detokenized: {:?}", detokenized);
        // println!("Num tokens: {:?}", tokens.len());
        // println!("Vocabulary size: {:?}", tokenizer.vocabulary.len());

        let expected = Tokenizer::clean_text(TEST_STR);
        let actual = detokenized;
        assert_eq!(actual.trim(), expected.trim());
    }

    fn has_duplicates(mut vocab: Vec<String>) -> bool {
        vocab.sort();
        vocab.windows(2).any(|w| w[0] == w[1])
    }  
}
