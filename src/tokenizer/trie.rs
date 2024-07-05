use std::collections::HashMap;
use serde::{Serialize, Deserialize};


#[derive(Default, Serialize, Deserialize, Debug, Clone)]
pub struct TrieNode {
    pub children: HashMap<char, TrieNode>,
    pub is_terminal: bool,
}

impl TrieNode {
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_terminal: false,
        }
    }

    // Insert a word into the trie
    pub fn insert(&mut self, word: &str) {
        let mut current = self;
        for c in word.chars() {
            current = current.children.entry(c).or_insert_with(TrieNode::new);
        }
        current.is_terminal = true;
    }

    // Find the longest prefix that is a valid token
    pub fn find_longest_prefix<'a>(&'a self, text: &'a str) -> Option<(usize, &str)> {
        let mut current = self;
        let mut last_valid_length = None;
        let mut chars = text.chars().enumerate();

        while let Some((index, c)) = chars.next() {
            if let Some(next_node) = current.children.get(&c) {
                current = next_node;
                if current.is_terminal {
                    last_valid_length = Some((index + 1, &text[0..index + 1]));
                }
            } else {
                break;
            }
        }
        last_valid_length
    }
}