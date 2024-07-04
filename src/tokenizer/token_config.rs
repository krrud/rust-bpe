use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Token {
    pub value: String,
    pub index: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenConfig {
    pub eow: Token,
    pub eos: Token,
    pub space: Token,
    pub pad: Token,
    pub mask: Token,
    pub unknown: Token,
}

impl TokenConfig {
    pub fn new() -> Self {
        TokenConfig {
            eow: Token { value: "</w>".to_string(), index: 0 },
            eos: Token { value: "</s>".to_string(), index: 1 },
            space: Token { value: " ".to_string(), index: 2 },
            pad: Token { value: "<pad>".to_string(), index: 3 },
            mask: Token { value: "<mask>".to_string(), index: 4 },
            unknown: Token { value: "<unk>".to_string(), index: 5 },
        }
    }

    pub fn get_values(&self) -> Vec<String> {
        vec![
            self.eow.value.clone(),
            self.eos.value.clone(),
            self.space.value.clone(),
            self.pad.value.clone(),
            self.mask.value.clone(),
            self.unknown.value.clone(),
        ]
    }

    pub fn get_indices(&self) -> Vec<usize> {
        vec![
            self.eow.index,
            self.eos.index,
            self.space.index,
            self.pad.index,
            self.mask.index,
            self.unknown.index,
        ]
    }

    pub fn is_eos(&self, token: &str) -> bool {
        token == self.eos.value
    }

    pub fn set_indices(&mut self, indices: Vec<usize>) {
        self.eow.index = indices[0];
        self.eos.index = indices[1];
        self.space.index = indices[2];
        self.pad.index = indices[3];
        self.mask.index = indices[4];
        self.unknown.index = indices[5];
    }
}
