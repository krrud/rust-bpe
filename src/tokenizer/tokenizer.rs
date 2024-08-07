use std::collections::{HashMap, HashSet};
use dashmap::DashMap;
use rayon::prelude::*;
use std::io::{self, Write};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use serde_json;

use crate::tokenizer::TokenConfig;
use crate::tokenizer::TrieNode;


#[derive(Serialize, Deserialize, Debug)]
pub struct Tokenizer {
    #[serde(skip_serializing, skip_deserializing)]
    pub vocabulary_trie: TrieNode,
    pub vocabulary: HashSet<String>,
    pub merge_rules: Vec<(String, String)>,
    pub token_to_index: HashMap<String, usize>,
    pub index_to_token: HashMap<usize, String>,
    pub config: TokenConfig,
}

impl Tokenizer {
    pub fn new(vocabulary: HashSet<String>, merge_rules: Vec<(String, String)>, config: TokenConfig) -> Self {
        let mut vocabulary_trie = TrieNode::new();
        let mut token_to_index = HashMap::new();
        let mut index_to_token = HashMap::new();
        
        for (i, token) in vocabulary.iter().enumerate() {
            token_to_index.insert(token.clone(), i);
            index_to_token.insert(i, token.clone());
            vocabulary_trie.insert(&token);
        }

        Tokenizer {
            vocabulary_trie,
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
    
    pub fn clean_text(text: &str) -> String {
        text.to_lowercase()
    }

    pub fn tokenize(&self, input_text: &str) -> Vec<usize> {
        let text = Self::clean_text(input_text);
        let mut tokens = Vec::new();
        let mut start = 0;
    
        while start < text.len() {
            if let Some((length, substr)) = self.vocabulary_trie.find_longest_prefix(&text[start..]) {
                // Add the longest match and update the start position
                tokens.push(self.get_index(substr).unwrap());
                start += length;
            } else {
                // If no token is found, append unknown token
                tokens.push(self.config.unknown.index);
                start += 1;
            }
        }
        tokens
    }

    pub fn detokenize(&self, indices: &[usize]) -> String {
        let mut result = String::new();
        let mut tokens = Vec::new();
    
        // Collect tokens from indices
        for &idx in indices.iter() {
            if let Some(token) = self.get_token(idx) {
                tokens.push(token);
            }
        }
    
        for i in 0..tokens.len() {
            let token = &tokens[i];
            let next_token = tokens.get(i + 1).cloned().unwrap_or_default();
    
            if !self.config.is_eos(token) {
                result.push_str(token);
            } else if !self.config.is_special_token(&next_token) {
                result.push(' ');
            }
        }
    
        result
    }

    pub fn train_cpu(source: &str, iterations: usize, output_filepath: &str, start_filepath: Option<&str>) -> Self {
        // Train tokenizer on CPU using byte pair encoding
        let start_time = Instant::now();
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

            // Populate the token_map efficiently
            for (value, &index) in values.iter().zip(indices.iter()) {
                token_map.insert(value.clone(), index);
            }

            // Utilize a buffer for character conversion to reduce allocation
            let mut char_buffer = String::with_capacity(4); // Capacity for a single character string

            for word in source.split_whitespace() {
                for c in word.chars() {
                    char_buffer.clear();
                    char_buffer.push(c);
                    char_buffer.make_ascii_lowercase();

                    let index = *token_map.entry(char_buffer.clone()).or_insert_with(|| {
                        let new_index = token_list.len();
                        token_list.push(char_buffer.clone());
                        new_index
                    });
                    token_indices.push(index);
                }
                token_indices.push(config.space.index);
            }

            // Remove the last space index if it exists
            if let Some(&last) = token_indices.last() {
                if last == config.space.index {
                    token_indices.pop();
                }
            }
        }
        println!("Init time: {:?}", start_time.elapsed().as_secs_f32());

        for i in 0..iterations {
            let iter_time = Instant::now();
            let pair_frequencies = DashMap::new();

            token_indices.par_chunks(8192).for_each(|chunk| {
                let mut local_map = HashMap::new();
                for window in chunk.windows(2) {
                    if let [a, b] = window {
                        *local_map.entry((*a, *b)).or_insert(0) += 1;
                    }
                }
                // Merge local map into the global DashMap
                for (key, count) in local_map {
                    *pair_frequencies.entry(key).or_insert(0) += count;
                }
            });
            
            // Aggregate results based on conditions
            let (first, second) = pair_frequencies.iter().max_by_key(|entry| *entry.value())
                .map(|entry| *entry.key())
                .unwrap_or((usize::MAX, usize::MAX));

            if first != usize::MAX && second != usize::MAX {
                let new_token = format!("{}{}", token_list[first], token_list[second]);
                let new_index = token_list.len();
                token_list.push(new_token);
                merge_rules.push((token_list[first].clone(), token_list[second].clone()));
            
                // Update token_indices
                let mut new_token_indices = Vec::with_capacity(token_indices.len());
                let mut i = 0;
                while i < token_indices.len() {
                    if i < token_indices.len() - 1 && token_indices[i] == first && token_indices[i + 1] == second {
                        new_token_indices.push(new_index);
                        i += 2; // Skip the next index - it's part of a merged pair
                    } else {
                        new_token_indices.push(token_indices[i]);
                        i += 1;
                    }
                }
                if i == token_indices.len() - 1 {
                    new_token_indices.push(token_indices[i]); // Push the last element if it wasn't part of a pair
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

            println!("Iteration {} time: {:?}", i, iter_time.elapsed().as_secs_f32());
            io::stdout().flush().unwrap();
        }
    
        let vocabulary: HashSet<String> = token_list.into_iter().collect();
        config.set_indices(Self::extract_indices(&vocabulary, &config));
        let trained_tokenizer = Tokenizer::new(vocabulary, merge_rules, config);
        trained_tokenizer.save(output_filepath).unwrap();
        println!("Total time: {:?}", start_time.elapsed().as_secs_f32());

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

    pub fn process_dataset(dir: &str) -> String {
        // Process all .txt files in the provided directory into a single string
        let mut text = String::new();
        let mut i = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let file = entry.unwrap();
            let path = file.path();
            if path.is_file() {
                println!("Reading file: {:?}", file.file_name());
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        let source_text = String::from_utf8_lossy(&bytes);
                        let clean_text = Tokenizer::clean_text(&source_text);
                        if i > 0 {
                            text.push_str(" ");
                        }
                        text.push_str(&clean_text);
                    },
                    Err(e) => {
                        eprintln!("Could not read file {:?}: {}", file.file_name(), e);
                        continue;
                    }
                }
            }
            i += 1;
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
        let mut tokenizer: Tokenizer = serde_json::from_str(&data)?;
        tokenizer.build_trie();

        Ok(tokenizer)
    }

    pub fn build_trie(&mut self) {
        let mut vocabulary_trie = TrieNode::new();
        for token in self.vocabulary.iter() {
            vocabulary_trie.insert(token);
        }
        self.vocabulary_trie = vocabulary_trie;
    }
}
