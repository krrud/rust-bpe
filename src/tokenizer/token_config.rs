use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Token {
    pub value: String,
    pub index: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenConfig {
    pub sot: Token,
    pub eot: Token,
    pub eos: Token,
    pub pad: Token,
    pub mask: Token,
    pub unknown: Token,
    pub space: Token,
    pub newline: Token,
    pub carriage: Token,
}

impl TokenConfig {
    pub fn new() -> Self {
        TokenConfig {
            sot: Token {value: "<|sot|>".to_string(), index: 0},
            eot: Token {value: "<|eot|>".to_string(), index: 1},
            eos: Token {value: "<|eos|>".to_string(), index: 2},
            pad: Token {value: "<pad>".to_string(), index: 3},
            mask: Token {value: "<mask>".to_string(), index: 4},
            unknown: Token {value: "<unk>".to_string(), index: 5},
            space: Token {value: " ".to_string(), index: 6},
            newline: Token {value: "\n".to_string(), index: 7},
            carriage: Token {value: "\r".to_string(), index: 8},
        }
    }

    pub fn get_values(&self) -> Vec<String> {
        vec![
            self.sot.value.clone(),
            self.eot.value.clone(),
            self.eos.value.clone(),
            self.pad.value.clone(),
            self.mask.value.clone(),
            self.unknown.value.clone(),
            self.space.value.clone(),
            self.newline.value.clone(),
            self.carriage.value.clone(),
        ]
    }

    pub fn get_indices(&self) -> Vec<usize> {
        vec![
            self.sot.index,
            self.eot.index,
            self.eos.index,
            self.pad.index,
            self.mask.index,
            self.unknown.index,
            self.space.index,
            self.newline.index,
            self.carriage.index,
        ]
    }

    pub fn is_eos(&self, token: &str) -> bool {
        token == self.eos.value
    }

    pub fn is_special_token(&self, token: &str) -> bool {
        self.get_values().contains(&token.to_string())
    }

    pub fn set_indices(&mut self, indices: Vec<usize>) {
        assert_eq!(indices.len(), 9, "Indices vector must have 9 elements.");
        self.sot.index = indices[0];
        self.eot.index = indices[1];
        self.eos.index = indices[2];
        self.pad.index = indices[3];
        self.mask.index = indices[4];
        self.unknown.index = indices[5];
        self.space.index = indices[6];
        self.newline.index = indices[7];
        self.carriage.index = indices[8];
    }
}
