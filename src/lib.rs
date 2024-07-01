mod tokenizer;
use tokenizer::Tokenizer;
use wasm_bindgen::prelude::*;
use std::collections::HashSet;


#[wasm_bindgen]
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl TokenizerWrapper {
    #[wasm_bindgen(constructor)]
    pub fn new(vocabulary: JsValue, merge_rules: JsValue) -> TokenizerWrapper {
        let vocabulary: HashSet<String> = vocabulary.into_serde().unwrap();
        let merge_rules: Vec<(String, String)> = merge_rules.into_serde().unwrap();
        TokenizerWrapper {
            tokenizer: Tokenizer::new(vocabulary, merge_rules)
        }
    }

    #[wasm_bindgen(getter, js_name = getVocabulary)]
    pub fn get_vocabulary(&self) -> JsValue {
        JsValue::from_serde(&self.tokenizer.get_vocabulary()).unwrap()
    }

    #[wasm_bindgen(getter, js_name = getMergeRules)]
    pub fn get_merge_rules(&self) -> JsValue {
        JsValue::from_serde(&self.tokenizer.get_merge_rules()).unwrap()
    }

    #[wasm_bindgen(js_name = getToken)]
    pub fn get_token(&self, index: usize) -> JsValue {
        JsValue::from_serde(&self.tokenizer.get_token(index)).unwrap()
    }

    #[wasm_bindgen(js_name = getIndex)]
    pub fn get_index(&self, token: &str) -> JsValue {
        JsValue::from_serde(&self.tokenizer.get_index(token)).unwrap()
    }

    #[wasm_bindgen(js_name = getTokens)]
    pub fn get_tokens(&self, indices: JsValue) -> JsValue {
        let indices: Vec<usize> = indices.into_serde().unwrap();
        JsValue::from_serde(&self.tokenizer.get_tokens(&indices)).unwrap()
    }

    #[wasm_bindgen(js_name = getIndices)]
    pub fn get_indices(&self, tokens: JsValue) -> JsValue {
        let tokens: Vec<String> = tokens.into_serde().unwrap();
        JsValue::from_serde(&self.tokenizer.get_indices(&tokens)).unwrap()
    }

    #[wasm_bindgen(js_name = tokenize)]
    pub fn tokenize(&self, text: &str) -> JsValue {
        JsValue::from_serde(&self.tokenizer.tokenize(text)).unwrap()
    }

    #[wasm_bindgen(js_name = detokenize)]
    pub fn detokenize(&self, indices: JsValue) -> String {
        let indices: Vec<usize> = indices.into_serde().unwrap();
        self.tokenizer.detokenize(&indices)
    }

    #[wasm_bindgen(js_name = cleanText)]
    pub fn clean_text(text: &str) -> String {
        Tokenizer::clean_text(text)
    }

    #[wasm_bindgen(js_name = save)]
    pub fn save(&self, path: &str) -> JsValue {
        JsValue::from_serde(&self.tokenizer.save(path).unwrap()).unwrap()
    }

    #[wasm_bindgen(js_name = load)]
    pub fn load(path: &str) -> JsValue {
        JsValue::from_serde(&Tokenizer::load(path).unwrap()).unwrap()
    }

    #[wasm_bindgen(js_name = padSequences)]
    pub fn pad_sequences(&self, sequences: JsValue, max_len: usize) -> JsValue {
        let sequences: Vec<Vec<usize>> = sequences.into_serde().unwrap();
        JsValue::from_serde(&self.tokenizer.pad_sequences(&sequences, max_len)).unwrap()
    }
}