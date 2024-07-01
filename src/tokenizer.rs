use std::collections::{HashMap, HashSet, VecDeque};
use rayon::prelude::*;
use std::io::stdout;
use std::io::{self, Write};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use serde_json;

use crate::gpu::{initialize_wgpu, prep_token_data, run_gpu_pipeline};


#[derive(Serialize, Deserialize, Debug)]
pub struct Tokenizer {
    vocabulary: HashSet<String>,
    merge_rules: Vec<(String, String)>,
    token_to_index: HashMap<String, usize>,
    index_to_token: HashMap<usize, String>,
}

impl Tokenizer {
    pub fn new(vocabulary: HashSet<String>, merge_rules: Vec<(String, String)>) -> Self {
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

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // Tokenize input text based on learned merge rules
        let cleaned_text = Tokenizer::clean_text(text);
        let mut words: Vec<Vec<String>> = cleaned_text.split_whitespace()
            .map(|word| {
                word.chars()
                    .map(|c| c.to_string().to_lowercase())
                    .collect::<Vec<String>>()
                    .into_iter()
                    .chain(std::iter::once("</w>".to_string()))
                    .collect()
            })
            .collect();
    
        // Apply merge rules
        for (first, second) in &self.merge_rules {
            let new_token = format!("{}{}", first, second);
            words = words.iter()
                .map(|tokens| {
                    let mut new_tokens = Vec::with_capacity(tokens.len());
                    let mut i = 0;
                    while i < tokens.len() {
                        if i + 1 < tokens.len() && &tokens[i] == first && &tokens[i + 1] == second {
                            new_tokens.push(new_token.clone());
                            i += 2; // skip next token since it's merged
                        } else {
                            new_tokens.push(tokens[i].clone());
                            i += 1;
                        }
                    }
                    new_tokens
                })
                .collect();
        }

        let unk_index = self.get_index("<unk>").unwrap_or(0);
        words.iter()
            .flat_map(|tokens| 
                tokens.iter().map(|token| 
                    self.get_index(&token).unwrap_or(unk_index)
                )
            )
            .collect()
    }


    pub fn detokenize(&self, indices: &[usize]) -> String {
        // Convert token indices back to text
        let mut result = String::new();
        let mut previous_token_was_word = false;

        for &idx in indices {
            if let Some(token) = self.get_token(idx) {
                if token.ends_with("</w>") {
                    // Add the token without the </w>, and a space if it's not the first word
                    if !result.is_empty() && previous_token_was_word {
                        result.push(' ');
                    }
                    result.push_str(&token.replace("</w>", ""));
                    previous_token_was_word = true;
                } else {
                    // Handle punctuation or other non-word tokens
                    if previous_token_was_word {
                        result.push(' ');
                    }
                    result.push_str(&token);
                    previous_token_was_word = false;
                }
            }
        }
        result
    }

    pub fn train_cpu(source: &str, iterations: usize, output_filepath: Option<&str>) -> Self {
        // Train tokenizer on CPU using byte pair encoding
        let filepath = output_filepath.unwrap_or("./src/tokenizer.json");
        let mut merge_rules: Vec<(String, String)> = Vec::new();
        let mut token_list: Vec<String> = vec!["</w>".to_string()];
        let mut token_indices: Vec<Vec<usize>> = source.split_whitespace()
            .map(|word| {
                word.chars()
                    .map(|c| {
                        let s = c.to_string().to_lowercase();
                        let index = token_list.iter().position(|x| x == &s).unwrap_or_else(|| {
                            token_list.push(s.clone());
                            token_list.len() - 1
                        });
                        index
                    })
                    .chain(std::iter::once(0)) // Add index for "</w>"
                    .collect::<Vec<_>>()
            })
            .collect();
        
        let start_time = Instant::now();
        for i in 0..iterations {
            // Count pair frequencies
            let pair_frequencies: HashMap<(usize, usize), usize> = token_indices.par_iter()
                .flat_map_iter(|tokens| {
                    tokens.windows(2).filter_map(|window| {
                        match window {
                            [a, b] => Some(((*a, *b), 1)),
                            _ => None,
                        }
                    })
                })
                .fold(|| HashMap::new(), |mut acc, (pair, count)| {
                    *acc.entry(pair).or_insert(0) += count;
                    acc
                })
                .reduce(HashMap::new, |mut acc, partial| {
                    for (key, count) in partial {
                        *acc.entry(key).or_insert(0) += count;
                    }
                    acc
                });

            // Find the most frequent pair and merge them
            if let Some(((first, second), _)) = pair_frequencies.par_iter().max_by_key(|&(_, &count)| count) {
                let new_token = format!("{}{}", token_list[*first], token_list[*second]);
                let new_index = token_list.len();
                token_list.push(new_token);
                merge_rules.push((token_list[*first].clone(), token_list[*second].clone()));
    
                token_indices.par_iter_mut().for_each(|tokens| {
                    let mut i = 0;
                    while i < tokens.len() {
                        if i + 1 < tokens.len() && tokens[i] == *first && tokens[i + 1] == *second {
                            tokens[i] = new_index;
                            tokens.remove(i + 1); // remove the second part of the merged pair
                        } else {
                            i += 1;
                        }
                    }
                });
            } else {
                break; // No more pairs to merge
            }
            
            // Save every 50 iterations
            if i % 50 == 0 {
                let vocabulary: HashSet<String> = token_list.clone().into_iter().collect();
                let tokenizer = Tokenizer::new(vocabulary, merge_rules.clone());
                tokenizer.save(&format!("{}", filepath)).unwrap();
            }

            // Log progress and metrics
            let elapsed_time = start_time.elapsed().as_secs_f32();
            let percent = 100.0 * ((i + 1) as f32 / iterations as f32);
            let iteration_time = elapsed_time / (i as f32 + 1.0);
            print!("\rTraining Tokenizer: {:.2}% | ETA: {:.2}s", percent, iteration_time * (iterations as f32 - (i as f32 + 1.0)));
            io::stdout().flush().unwrap();
        }

        // Create tokenizer from final vocabulary and merge rules
        let vocabulary: HashSet<String> = token_list.into_iter().collect();
        Tokenizer::new(vocabulary, merge_rules)
    }

    pub async fn train_gpu(source: &str, iterations: usize, vocab_size: usize, output_filepath: Option<&str>) -> Self {
        // Train tokenizer on GPU using byte pair encoding
        let filepath = output_filepath.unwrap_or("./src/tokenizer.json");
        let mut merge_rules: Vec<(String, String)> = Vec::new();
        let mut token_list: Vec<String> = vec!["</w>".to_string()];
        let mut token_map: HashMap<String, u32> = HashMap::new();
        token_map.insert("</w>".to_string(), 0);
    
        // Initialize token indices
        let mut token_indices: Vec<u32> = source.split_whitespace()
            .flat_map(|word| {
                word.chars()
                    .map(|c| {
                        let s = c.to_string().to_lowercase();
                        if let Some(&index) = token_map.get(&s) {
                            index
                        } else {
                            let index = token_list.len() as u32;
                            token_list.push(s.clone());
                            token_map.insert(s, index);
                            index
                        }
                    })
                    .chain(std::iter::once(0)) // Add index for "</w>"
                    .collect::<Vec<u32>>()
            })
            .collect();
    
        let start_time = Instant::now();
        let (device, queue) = initialize_wgpu().await;

        for i in 0..iterations {
            let iter_time = Instant::now();
            let pair_frequencies_gpu = run_gpu_pipeline(&device, &queue, &token_indices, 128).await;
            let max_pair = pair_frequencies_gpu.par_iter().max_by_key(|&(_, &count)| count);

            if let Some(((first, second), _)) = max_pair {
                let first = *first as u32;
                let second = *second as u32;
                let new_token = format!("{}{}", token_list[first as usize], token_list[second as usize]);
                let new_index = token_list.len() as u32;
                token_list.push(new_token.clone());
                merge_rules.push((token_list[first as usize].clone(), token_list[second as usize].clone()));
    
                let update_time = Instant::now();
                let mut new_token_indices = Vec::with_capacity(token_indices.len());
                let mut iter = token_indices.iter().peekable();
                while let Some(&current) = iter.next() {
                    if iter.peek() == Some(&&second) && current == first {
                        new_token_indices.push(new_index);
                        iter.next(); // Skip the second token in the pair
                    } else {
                        new_token_indices.push(current);
                    }
                }
                token_indices = new_token_indices;
    
            } else {
                break; // No more pairs to merge
            }
    
            // Save tokenizer every 50 iterations
            if i % 50 == 0 {
                let vocabulary: HashSet<String> = token_list.clone().into_iter().collect();
                let tokenizer = Tokenizer::new(vocabulary, merge_rules.clone());
                tokenizer.save(&format!("{}", filepath)).unwrap();

                // Log progress and metrics
                let elapsed_time = start_time.elapsed().as_secs_f32();
                let percent = 100.0 * ((i + 1) as f32 / iterations as f32);
                let iteration_time = elapsed_time / (i as f32 + 1.0);
                print!("\rTraining Tokenizer: {:.2}% | ETA: {:.2}s", percent, iteration_time * (iterations as f32 - (i as f32 + 1.0)));
                io::stdout().flush().unwrap();
            }
            println!("Iteration {} time: {:?}", i, iter_time.elapsed().as_secs_f32());
        }
    
        let vocabulary: HashSet<String> = token_list.into_iter().collect();
        Tokenizer::new(vocabulary, merge_rules)
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
                        text.push(' '); // space between files
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
        let tokenizer: Tokenizer = serde_json::from_str(&data)?;
        Ok(tokenizer)
    }

    pub fn pad_sequences(&self, sequences: &Vec<Vec<usize>>, max_len: usize) -> Vec<Vec<usize>> {
        // Pad sequences to a fixed length
        let pad_token_id = 0;
        sequences.iter().map(|seq| {
            if seq.len() > max_len {
                // Truncate sequence if it's too long
                seq[..max_len].to_vec()
            } else {
                // Pad sequence with pad_token_id if it's too short
                let mut padded_seq = seq.clone();
                padded_seq.resize(max_len, pad_token_id);
                padded_seq
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_STR: &str = "The quick brown fox jumps over the lazy dog, while pack_my_box_with_five_dozen_liquor_jugs! Testing, 1234... How's that? #Tokenizer-Challenge!";

    #[test]
    fn validate_tokenizer() {
        let tokenizer = Tokenizer::load("./src/tokenizer_13k.json").unwrap();
        println!("Vocabulary size: {:?}", tokenizer.vocabulary.len());

        let duplicates = has_duplicates(tokenizer.vocabulary.iter().cloned().collect::<Vec<_>>());
        assert_eq!(duplicates, false);

        let tokens = tokenizer.tokenize(&TEST_STR);
        let token_vals = tokenizer.get_tokens(&tokens);
        let token_indices = tokenizer.get_indices(&token_vals);
        println!("Tokens: {:?}", token_vals); 
        println!("Token indices: {:?}", token_indices);

        let detokenized = tokenizer.detokenize(&tokens);
        println!("Detokenized: {:?}", detokenized);
        
        let expected = Tokenizer::clean_text(TEST_STR);
        let actual = detokenized;
        assert_eq!(actual.trim(), expected.trim());
    }

    fn has_duplicates(mut vocab: Vec<String>) -> bool {
        vocab.sort();
        vocab.windows(2).any(|w| w[0] == w[1])
    }
    
}
