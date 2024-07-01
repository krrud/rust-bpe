use std::collections::{HashMap, HashSet, VecDeque};
use rayon::prelude::*;
use std::io::stdout;
use std::io::{self, Write};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use serde_json;


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

    pub fn train_cpu_single_word(source: &str, iterations: usize, output_filepath: Option<&str>) -> Self {
        // Train tokenizer on CPU using byte pair encoding - only capable of handling single word tokens
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
            let iter_time = Instant::now();
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
            // print!("\rTraining Tokenizer: {:.2}% | ETA: {:.2}s", percent, iteration_time * (iterations as f32 - (i as f32 + 1.0)));
            println!("Iteration {} time: {:?}", i, iter_time.elapsed().as_secs_f32());
            io::stdout().flush().unwrap();
        }

        // Create tokenizer from final vocabulary and merge rules
        let vocabulary: HashSet<String> = token_list.into_iter().collect();
        Tokenizer::new(vocabulary, merge_rules)
    }

    pub fn train_cpu(source: &str, iterations: usize, output_filepath: Option<&str>) -> Self {
        let filepath = output_filepath.unwrap_or("./src/tokenizer.json");
        let mut merge_rules: Vec<(String, String)> = Vec::new();
        let mut token_list: Vec<String> = vec!["</w>".to_string(), " ".to_string()];
        let end_of_word_idx = 0;
        let space_idx = 1;
    
        // Split the source into words and handle spaces between words
        let mut token_indices: Vec<usize> = Vec::with_capacity(source.len());
        let mut token_map: HashMap<String, usize> = HashMap::new();
        token_map.insert("</w>".to_string(), end_of_word_idx);
        token_map.insert(" ".to_string(), space_idx);
    
        for word in source.split_whitespace() {
            for c in word.chars() {
                let s = c.to_string().to_lowercase();
                let index = *token_map.entry(s.clone()).or_insert_with(|| {
                    token_list.push(s);
                    token_list.len() - 1
                });
                token_indices.push(index);
            }
            token_indices.push(end_of_word_idx); // End of word token
            token_indices.push(space_idx); // Space between words
        }
        if *token_indices.last().unwrap() == space_idx {
            token_indices.pop();
        }
    
        let start_time = Instant::now();
        for i in 0..iterations {
            let iter_time = Instant::now();
    
            // Count pair frequencies using Rayon for parallel processing
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
    
            // Delay merging common pairs in the first few iterations to avoid bottlenecks
            let delay_common_pairs = 10;
            let should_delay = |token: &String| token == "</w>" || token == " ";
    
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
                // println!("Merging: '{}' + '{}' into '{}'", token_list[first], token_list[second], new_token);
    
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
                break; // No more pairs to merge
            }
    
            // Save every 50 iterations
            if i % 50 == 0 {
                let vocabulary: HashSet<String> = token_list.clone().into_iter().collect();
                let tokenizer = Tokenizer::new(vocabulary, merge_rules.clone());
                tokenizer.save(filepath).unwrap();
            }
    
            let elapsed_time = start_time.elapsed().as_secs_f32();
            let percent = 100.0 * ((i + 1) as f32 / iterations as f32);
            let iteration_time = elapsed_time / (i as f32 + 1.0);
            println!("Iteration {} time: {:?}", i, iter_time.elapsed().as_secs_f32());
            io::stdout().flush().unwrap();
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
        let tokenizer = Tokenizer::load("./src/tokenizer_training.json").unwrap();
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
