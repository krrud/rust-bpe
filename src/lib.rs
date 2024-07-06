pub mod tokenizer;
use tokenizer::{Tokenizer, TokenConfig};

use std::collections::{HashMap, HashSet};
use wasm_bindgen::prelude::*;
use serde_wasm_bindgen::{to_value, from_value};

#[cfg(not(target_arch = "wasm32"))]
use pyo3::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use pyo3::exceptions::{PyValueError, PyIOError};



// JavaScript API
#[wasm_bindgen]
pub struct TokenizerJs {
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl TokenizerJs {
    #[wasm_bindgen(constructor)]
    pub fn new(vocabulary: JsValue, merge_rules: JsValue, config: JsValue) -> TokenizerJs {
        let vocabulary: HashSet<String> = from_value(vocabulary).unwrap();
        let merge_rules: Vec<(String, String)> = from_value(merge_rules).unwrap();
        let config = from_value(config).unwrap_or(TokenConfig::new());
        TokenizerJs {
            tokenizer: Tokenizer::new(vocabulary, merge_rules, config)
        }
    }

    #[wasm_bindgen(getter, js_name = getVocabulary)]
    pub fn get_vocabulary(&self) -> JsValue {
        to_value(&self.tokenizer.get_vocabulary()).unwrap()
    }

    #[wasm_bindgen(getter, js_name = getMergeRules)]
    pub fn get_merge_rules(&self) -> JsValue {
        to_value(&self.tokenizer.get_merge_rules()).unwrap()
    }

    #[wasm_bindgen(js_name = getToken)]
    pub fn get_token(&self, index: usize) -> JsValue {
        to_value(&self.tokenizer.get_token(index)).unwrap()
    }

    #[wasm_bindgen(js_name = getIndex)]
    pub fn get_index(&self, token: &str) -> JsValue {
        to_value(&self.tokenizer.get_index(token)).unwrap()
    }

    #[wasm_bindgen(js_name = getTokens)]
    pub fn get_tokens(&self, indices: JsValue) -> JsValue {
        let indices: Result<Vec<usize>, _> = from_value(indices);
        match indices {
            Ok(indices) => {
                let tokens = self.tokenizer.get_tokens(&indices);
                to_value(&tokens).unwrap_or_else(|e| to_value(&format!("Failed to serialize tokens: {}", e)).unwrap())
            },
            Err(e) => to_value(&format!("Failed to deserialize indices: {}", e)).unwrap()
        }
    }

    #[wasm_bindgen(js_name = getIndices)]
    pub fn get_indices(&self, tokens: JsValue) -> JsValue {
        let tokens: Vec<String> = from_value(tokens).unwrap();
        to_value(&self.tokenizer.get_indices(&tokens)).unwrap()
    }

    #[wasm_bindgen(js_name = tokenize)]
    pub fn tokenize(&self, text: &str) -> JsValue {
        to_value(&self.tokenizer.tokenize(text)).unwrap()
    }

    #[wasm_bindgen(js_name = detokenize)]
    pub fn detokenize(&self, indices: JsValue) -> String {
        let indices: Vec<usize> = from_value(indices).unwrap();
        self.tokenizer.detokenize(&indices)
    }

    #[wasm_bindgen(js_name = cleanText)]
    pub fn clean_text(text: &str) -> String {
        Tokenizer::clean_text(text)
    }

    #[wasm_bindgen(js_name = save)]
    pub fn save(&self, path: &str) -> JsValue {
        match self.tokenizer.save(path) {
            Ok(result) => to_value(&result).unwrap(),
            Err(e) => to_value(&format!("Error saving tokenizer: {}", e)).unwrap(),
        }
    }

    #[wasm_bindgen(js_name = load)]
    pub fn load(path: &str) -> JsValue {
        match Tokenizer::load(path) {
            Ok(tokenizer) => to_value(&tokenizer).unwrap(),
            Err(e) => to_value(&format!("Error loading tokenizer: {}", e)).unwrap(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pyclass]
struct TokenizerPy {
    tokenizer: Tokenizer,
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl TokenizerPy {
    #[new]
    fn new(config_path: &str) -> PyResult<Self> {
        // Read and parse the JSON configuration file
        let data = std::fs::read_to_string(config_path)
            .map_err(|e| PyErr::new::<PyIOError, _>(format!("Error reading config file: {}", e)))?;
        
        let config: HashMap<String, serde_json::Value> = serde_json::from_str(&data)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Error parsing JSON: {}", e)))?;

        // Extract and convert JSON values
        let vocabulary: HashSet<String> = serde_json::from_value(config.get("vocabulary").cloned().unwrap_or_default())
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Error parsing vocabulary: {}", e)))?;
        let merge_rules: Vec<(String, String)> = serde_json::from_value(config.get("merge_rules").cloned().unwrap_or_default())
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Error parsing merge rules: {}", e)))?;
        let config = serde_json::from_value(config.get("config").cloned().unwrap_or_default())
            .unwrap_or_else(|_| TokenConfig::new());

        Ok(TokenizerPy {
            tokenizer: Tokenizer::new(vocabulary, merge_rules, config),
        })
    }
    
    #[getter]
    fn get_vocabulary(&self) -> PyResult<HashSet<String>> {
        Ok(self.tokenizer.get_vocabulary())
    }

    #[getter]
    fn get_merge_rules(&self) -> PyResult<Vec<(String, String)>> {
        Ok(self.tokenizer.get_merge_rules())
    }

    fn get_token(&self, index: usize) -> PyResult<String> {
        match self.tokenizer.get_token(index) {
            Some(index) => Ok(index),
            None => Err(PyErr::new::<PyValueError, _>(format!("Token not found: {}", index))),
        }
    }

    fn get_index(&self, token: &str) -> PyResult<usize> {
        match self.tokenizer.get_index(token) {
            Some(index) => Ok(index),
            None => Err(PyErr::new::<PyValueError, _>(format!("Token not found: {}", token))),
        }
    }

    fn get_tokens(&self, indices: Vec<usize>) -> PyResult<Vec<String>> {
        Ok(self.tokenizer.get_tokens(&indices))
    }
    

    fn get_indices(&self, tokens: Vec<String>) -> PyResult<Vec<usize>> {
        Ok(self.tokenizer.get_indices(&tokens))
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<usize>> {
        Ok(self.tokenizer.tokenize(text))
    }

    fn detokenize(&self, indices: Vec<usize>) -> PyResult<String> {
        Ok(self.tokenizer.detokenize(&indices))
    }

    #[staticmethod]
    fn clean_text(text: &str) -> String {
        Tokenizer::clean_text(text)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.tokenizer.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving tokenizer: {}", e)))
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let tokenizer = Tokenizer::load(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading tokenizer: {}", e)))?;
        Ok(TokenizerPy { tokenizer })
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymodule]
fn rust_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TokenizerPy>()?;
    Ok(())
}